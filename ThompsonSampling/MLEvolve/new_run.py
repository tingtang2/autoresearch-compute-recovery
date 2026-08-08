"""Scheduler entrypoint.

CHANGES vs run.py:
- Phase-1 draft generation refactored into a reusable `run_phase1_drafts()` closure so
  it can be invoked again after a livelock restart.
- After each completed future, the scheduler now checks `agent.should_restart`. If set:
    1. Drain in-flight futures (with `interpreter.terminate_all_subprocesses()` to avoid
       hanging on subprocess work).
    2. Wait briefly for the consultant's daemon learning threads to commit any in-flight
       BANNED rules into `_global_conditional_rules`.
    3. Call `agent.prepare_for_restart()` to quiesce dead drafts and reset counters
       (BugConsultant is preserved — its accumulated rules are auto-injected into the
       new drafts via `draft_agent.run` → `get_prevention_guidance`).
    4. Re-run Phase-1 and submit the new drafts.
- Added a safety break when `agent.virtual_root.is_terminal` so the all-attempts-
  exhausted path actually exits instead of spinning on the terminal node.
"""

import atexit
import logging
import sys
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from utils.logging_config import setup_logging
import torch



def run():
    cfg = load_cfg()
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

    lock = threading.Lock()
    completed = 0

    # ── REFACTORED: Phase-1 is now a closure so it can be invoked on restart ──────────
    def run_phase1_drafts(label: str = "initial"):
        """Sequential Phase-1 draft generation (code only). Returns the pending draft nodes.

        Called once at startup, and again from the restart hook below. On restart,
        BANNED rules learned by the BugConsultant are automatically injected into
        the draft prompt via draft_agent.run -> get_prevention_guidance.
        """
        nodes = []
        remaining = total_steps - completed
        if initial_draft_count <= 0 or remaining <= 0:
            return nodes

        n_drafts = min(initial_draft_count, remaining)
        logger.info(f"📝 Phase 1 ({label}): Sequential draft generation (code only, {n_drafts} drafts)")

        def step_task_generate_only():
            logger.info(f"[step_task_generate_only] Generating draft from virtual root ({label})")
            return agent.step(exec_callback=exec_callback, node=None, execute_immediately=False)

        for draft_idx in range(n_drafts):
            try:
                logger.info(f"🔨 Generating draft {draft_idx + 1}/{n_drafts} (code only, {label})")
                cur_node = step_task_generate_only()
                nodes.append(cur_node)
                logger.info(f"✅ Draft {draft_idx + 1} code generated: node.id={cur_node.id}")
            except Exception as e:
                logger.exception(f"❌ Exception during draft {draft_idx + 1} generation: {e}")

        logger.info(f"✅ Phase 1 ({label}) complete: {len(nodes)} draft codes generated")
        return nodes

    pending_draft_nodes = run_phase1_drafts(label="initial")

    if pending_draft_nodes or completed < total_steps:
        logger.info(f"🚀 Phase 2: Pipelined parallel execution")
        logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
        logger.info(f"   - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            try:
                executed_node = agent.execute_deferred_node(node, exec_callback)
                logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                return None

        # ── NEW: restart hook helper ────────────────────────────────────────────────
        def perform_restart(executor, futures):
            """Drain in-flight work, wait for consultant to commit rules, re-run Phase 1.

            Returns the list of newly-submitted future objects (already added to `futures`).
            """
            attempt = agent.restart_attempts + 1
            logger.warning(
                f"[restart] Livelock detected. Attempt {attempt}/{agent.max_restart_attempts}. "
                f"Terminating subprocesses and draining {len(futures)} in-flight futures…"
            )

            # 1. Force any in-flight user-code subprocesses to abort so drain doesn't hang.
            try:
                interpreter.terminate_all_subprocesses()
            except Exception as kill_err:
                logger.warning(f"[restart] terminate_all_subprocesses warning: {kill_err}")

            # 2. Drain remaining futures with a per-future cap.
            for pending in list(futures):
                try:
                    pending.result(timeout=120)
                except Exception as drain_err:
                    logger.warning(f"[restart] drained future raised: {drain_err}")
                finally:
                    futures.discard(pending)
            logger.info(f"[restart] all in-flight futures drained")

            # 3. Wait briefly for the daemon consultant threads to land any in-flight
            #    BANNED rule writes into _global_conditional_rules.
            if agent.bug_consultant:
                deadline = time.time() + 30
                last_count = -1
                while time.time() < deadline:
                    rules_now = len(getattr(agent.bug_consultant, "_global_conditional_rules", []) or [])
                    if rules_now > 0 and rules_now == last_count:
                        break
                    last_count = rules_now
                    time.sleep(2)
                logger.info(
                    f"[restart] BugConsultant has "
                    f"{len(getattr(agent.bug_consultant, '_global_conditional_rules', []) or [])} "
                    f"global BANNED rules ready for injection."
                )
            else:
                logger.info("[restart] BugConsultant disabled; restart will rely on prompt resampling alone.")

            # 4. Quiesce dead drafts and reset virtual_root state.
            agent.prepare_for_restart()

            # 5. Re-run Phase 1; submit new drafts the same way the initial pass did.
            new_drafts = run_phase1_drafts(label=f"restart_{attempt}")
            submitted = []
            for i, node in enumerate(new_drafts):
                fut = executor.submit(execute_draft_node, node)
                futures.add(fut)
                submitted.append(fut)
                logger.info(f"📤 Submitted draft execution (restart {attempt}): {node.id}")
                if i < len(new_drafts) - 1:
                    time.sleep(10)
            return submitted

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = set()
            for i, node in enumerate(pending_draft_nodes):
                futures.add(executor.submit(execute_draft_node, node))
                logger.info(f"📤 Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
                    time.sleep(10)
                    logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

            initial_step_tasks = min(max_workers, total_steps - completed) - len(pending_draft_nodes)
            if initial_step_tasks > 0:
                for _ in range(initial_step_tasks):
                    futures.add(executor.submit(step_task))
                    logger.info(f"📤 Submitted initial step_task to fill thread pool")

            while completed < total_steps:
                # ── NEW: terminal-state safety break ────────────────────────────────
                if agent.virtual_root.is_terminal and not futures:
                    logger.error(
                        "[scheduler] virtual_root.is_terminal=True and no futures pending; "
                        "search has exhausted its restart budget. Exiting loop."
                    )
                    break

                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                if not done:
                    continue  # timeout, no completed futures, retry (allows SIGINT handling)

                for fut in done:
                    futures.remove(fut)
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                        else:
                            logger.warning(f"⚠️  Task returned None (execution failed)")
                    except Exception as e:
                        logger.exception(f"❌ Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1  # Exclude virtual node
                        if completed == total_steps:
                            logger.info(journal_to_string_tree(journal))

                    # ── NEW: restart hook ───────────────────────────────────────────
                    if agent.should_restart:
                        perform_restart(executor, futures)
                        # Skip the normal resubmit this iteration — the restart
                        # already filled the future pool with fresh drafts.
                        continue

                    # ── NEW: terminal-state safety break (inside loop) ──────────────
                    if agent.virtual_root.is_terminal:
                        logger.error(
                            "[scheduler] virtual_root.is_terminal=True; stopping submissions."
                        )
                        continue  # don't submit new work; outer-loop check will exit when futures drain

                    if completed + len(futures) < total_steps:
                        futures.add(executor.submit(step_task, cur_node))
                        logger.info(f"📤 Submitted next task based on node {cur_node.id if cur_node else 'None'}")
                    logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            interpreter.terminate_all_subprocesses()
            executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
            raise
        finally:
            if not interrupted:
                executor.shutdown(wait=True)
    else:
        logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

    interpreter.cleanup_session(-1)


if __name__ == "__main__":
    run()
