import atexit
import logging
import os
import shutil
import signal
import sys
import time as time_module

import backend
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

# Global shutdown event — threads check this to exit spin loops
_shutdown_event = threading.Event()
from agent.mcts_agent import MCTSAgent as Agent
from interpreter.interpreter_parallel import Interpreter
from search.journal import Journal
from search.node import Node
from omegaconf import OmegaConf
from rich.columns import Columns
from rich.console import Group
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
from rich.markdown import Markdown
from rich.status import Status
from rich.tree import Tree
from utils.config_mcts import load_task_desc, prep_agent_workspace, save_run, load_cfg

class VerboseFilter(logging.Filter):
    """
    Filter (remove) logs that have verbose attribute set to True
    """

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)


def journal_to_rich_tree(journal: Journal):
    best_node = journal.get_best_node()

    def append_rec(node: Node, tree):
        if node.is_buggy:
            s = "[red]◍ bug"
        else:
            style = "bold " if node is best_node else ""

            if node is best_node:
                s = f"[{style}green]● {node.metric.value:.3f} (best)"
            else:
                s = f"[{style}green]● {node.metric.value:.3f}"

        subtree = tree.add(s)
        for child in node.children:
            append_rec(child, subtree)

    tree = Tree("[bold blue]Solution tree")
    for n in journal.draft_nodes:
        append_rec(n, tree)
    return tree


def journal_to_string_tree(journal: Journal) -> str:
    best_node = journal.get_best_node()
    tree_str = "Solution tree\n"

    def append_rec(node: Node, level: int):
        nonlocal tree_str
        indent = "  " * level
        if node.is_buggy:
            s = f"{indent}◍ bug (ID: {node.id})\n"
        else:
            # support for multiple markers; atm only "best" is supported
            markers = []
            if node is best_node:
                markers.append("best")
            marker_str = " & ".join(markers)
            if marker_str and node.metric.value:
                s = f"{indent}● {node.metric.value:.3f} ({marker_str}) (ID: {node.id})\n"
            else:
                s = f"{indent}● {node.metric.value:.3f} (ID: {node.id})\n"
        tree_str += s
        for child in node.children:
            append_rec(child, level + 1)

    for n in journal.draft_nodes:
        append_rec(n, 0)

    return tree_str


def run():
    cfg = load_cfg()
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    # dont want info logs from httpx
    httpx_logger: logging.Logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("ml-master")
    # save logs to files as well, using same format
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # we'll have a normal log file and verbose log file. Only normal to console
    file_handler = logging.FileHandler(cfg.log_dir / "ml-master.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "ml-master.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    if cfg.agent.steerable_reasoning == True:
        logger.warning("Steerable reasoning is enabled, please make sure your open sourced model api support `client.compeletion.create()`, otherwise the process may fail")
        if "gpt" in cfg.agent.code.model or "gemini" in cfg.agent.code.model or "claude" in cfg.agent.code.model:
            logger.warning("Steerable reasoning does not support close sourced models, please set steerable reasoning to false")
            raise ValueError("Steerable reasoning does not support close sourced models, please set steerable reasoning to false")
    
    if cfg.agent.check_format == True:
        logger.warning("Check format is enabled, please make sure you have launched the server, or this step will be skipped")


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
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(exec_callback=exec_callback, node=node)
    
    max_workers = cfg.agent.search.parallel_search_num
    total_steps = cfg.agent.steps
    time_limit = cfg.agent.time_limit
    loop_start_time = time_module.time()

    # Handle SIGTERM — save and force-exit since ThreadPoolExecutor blocks shutdown
    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, saving and force-exiting...")
        _shutdown_event.set()
        try:
            save_run(cfg, journal)
        except Exception:
            pass
        os._exit(0)
    signal.signal(signal.SIGTERM, _sigterm_handler)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(step_task) for _ in range(min(max_workers, total_steps))}
    completed = 0
    lock = threading.Lock()
    try:
        while completed <= total_steps:
            # Graceful time check — exit 5 min before hard timeout for save + submission copy
            elapsed = time_module.time() - loop_start_time
            if elapsed > time_limit - 300:
                logger.warning(f"Time limit approaching ({elapsed:.0f}s / {time_limit}s). Saving and exiting gracefully.")
                with lock:
                    save_run(cfg, journal)
                    logger.info(journal_to_string_tree(journal))
                break

            done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=60)
            if not done:
                continue  # re-check time on wait timeout

            for fut in done:
                futures.remove(fut)
                try:
                    cur_node = fut.result()
                    logger.info(f"current node count is {completed}, current node.id is {cur_node.id}")
                except Exception as e:
                    logger.exception(f"Exception during step_task execution: {e}")
                    cur_node = None

                with lock:
                    save_run(cfg, journal)
                    completed = len(journal)-1  # Exclude virtual node
                    if completed == total_steps:
                        logger.info(journal_to_string_tree(journal))

                if completed + len(futures) < total_steps:
                    futures.add(executor.submit(step_task, cur_node))
    finally:
        # Signal workers to stop, then force-exit to avoid shutdown(wait=True) deadlock
        logger.info("Main loop exited. Signaling shutdown and cleaning up...")
        _shutdown_event.set()
        for f in futures:
            f.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        try:
            interpreter.cleanup_session(-1)
        except Exception:
            pass
        logger.info("Cleanup complete. Exiting process.")
        os._exit(0)


if __name__ == "__main__":    
    run()
