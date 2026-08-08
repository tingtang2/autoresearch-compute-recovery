import atexit
import logging
import os
import random
import shutil
import time

import numpy as np

from . import backend

from .agent import Agent
from .interpreter import ExecutionResult, Interpreter
from .journal import Journal, Node
from .journal2report import journal2report
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.status import Status
from rich.tree import Tree
from .utils.config import load_task_desc, prep_agent_workspace, save_run, load_cfg, export_final_submissions

logger = logging.getLogger("aide")


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        node_num = getattr(node, 'step', '?')
        if node.is_buggy:
            s = f"[red]#{node_num} ◍ bug"
        else:
            style = "bold " if node is best_node else ""

            # Include cv_std if available
            cv_std = getattr(node, 'cv_std', None)
            if cv_std is not None and cv_std > 0:
                std_str = f"±{cv_std:.3f}"
            else:
                std_str = ""

            if node is best_node:
                s = f"[{style}green]#{node_num} ● {node.metric.value:.3f}{std_str} (best)"
            else:
                s = f"[{style}green]#{node_num} ● {node.metric.value:.3f}{std_str}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def run():
    cfg = load_cfg()
    logger.info(f'Starting run "{cfg.exp_name}"')

    global_step = 0

    seed_env = os.environ.get("AIDE_SEED")
    if seed_env is not None:
        try:
            seed = int(seed_env)
        except ValueError:
            raise ValueError(f"Invalid AIDE_SEED={seed_env!r}; expected an int.")
        random.seed(seed)
        np.random.seed(seed)

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    # Try to resume from existing journal if it exists
    journal_path = cfg.log_dir / "journal.json"
    if journal_path.exists():
        from .utils import serialize
        print(f"[Resume] Found existing journal at {journal_path}")
        journal = serialize.load_json(journal_path, Journal)
        print(f"[Resume] Loaded {len(journal)} existing nodes, continuing from step {len(journal)}")
    else:
        journal = Journal()
        print(f"[New Run] Starting fresh journal")

    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),  # type: ignore
    )

    # Setup per-step grading if enabled
    grading_callback = None
    if cfg.per_step_grading.enabled:
        from .utils.mlebench_grading import setup_per_step_grading
        competition_id = cfg.competition_id or os.environ.get("COMPETITION_ID")
        grading_callback = setup_per_step_grading(cfg, competition_id)

    global_step = len(journal)
    prog = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    )
    status = Status("[green]Generating code...")
    prog.add_task("Progress:", total=cfg.agent.steps, completed=global_step)

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        max_retries = int(os.environ.get("AIDE_REPL_RETRIES", "2") or "2")
        for attempt in range(max_retries + 1):
            try:
                res = interpreter.run(*args, **kwargs)
                break
            except RuntimeError as e:
                # The REPL is a separate OS process; it can die (OOM/segfault/kill) without a Python exception.
                # On these failures, restart the REPL and retry this same step a few times.
                if "REPL child process" not in str(e):
                    raise
                logger.error(
                    "REPL crash at step %s (attempt %s/%s): %s",
                    len(journal),
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                try:
                    interpreter.cleanup_session()
                except Exception as cleanup_err:
                    logger.error(f"Cleanup after REPL crash failed: {cleanup_err}")
                if attempt < max_retries:
                    # Small backoff to avoid immediate crash loops (e.g., transient resource pressure).
                    time.sleep(0.5 * (attempt + 1))
                    continue
                res = ExecutionResult(
                    term_out=[f"RuntimeError: {e}\n"],
                    exec_time=0.0,
                    exc_type="RuntimeError",
                    exc_info={"args": [str(e)]},
                    exc_stack=None,
                )
        status.update("[green]Generating code...")
        return res

    def generate_live():
        tree = journal_to_rich_tree(journal)
        prog.update(prog.task_ids[0], completed=global_step)

        file_paths = [
            f"Result visualization:\n[yellow]▶ {str((cfg.log_dir / 'tree_plot.html'))}",
            f"Agent workspace directory:\n[yellow]▶ {str(cfg.workspace_dir)}",
            f"Experiment log directory:\n[yellow]▶ {str(cfg.log_dir)}",
        ]
        left = Group(
            Panel(Text(task_desc_str.strip()), title="Task description"), prog, status
        )
        right = tree
        wide = Group(*file_paths)

        return Panel(
            Group(
                Padding(wide, (1, 1, 1, 1)),
                Columns(
                    [Padding(left, (1, 2, 1, 1)), Padding(right, (1, 1, 1, 2))],
                    equal=True,
                ),
            ),
            title=f'[b]AIDE is working on experiment: [bold green]"{cfg.exp_name}[/b]"',
            subtitle="Press [b]Ctrl+C[/b] to stop the run",
        )

    with Live(
        generate_live(),
        refresh_per_second=16,
        screen=True,
    ) as live:
        try:
            while global_step < cfg.agent.steps:
                agent.step(exec_callback=exec_callback)
                save_run(cfg, journal)
                global_step = len(journal)

                # Per-step grading
                if grading_callback is not None:
                    grading_callback.on_step_complete(journal, global_step, cfg.workspace_dir, cfg)

                live.update(generate_live())
        finally:
            # Ensure cleanup happens even if loop exits early
            try:
                interpreter.cleanup_session()
            except Exception as e:
                logger.error(f"Final interpreter cleanup failed: {e}")

    # Export final submissions
    print("\nExporting final submissions...")
    export_final_submissions(cfg, journal)

    # Save per-step grading results
    if grading_callback is not None:
        print("Saving per-step grading results...")
        grading_callback.save_results()

    if cfg.generate_report:
        print("Generating final report from journal...")
        report = journal2report(journal, task_desc, cfg.report)
        print(report)
        report_file_path = cfg.log_dir / "report.md"
        with open(report_file_path, "w") as f:
            f.write(report)
        print("Report written to file:", report_file_path)


if __name__ == "__main__":
    run()
