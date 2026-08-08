"""run_backtrack — entrypoint for AIDE's MCTS / Thompson-Sampling backtracking agent.

The stock ``aide`` entrypoint (``aide.run:run``) uses ``aide.agent.Agent``, which has
no MCTS support, so passing ``agent.search.use_mcts=true`` to it does nothing. This
entrypoint instead uses ``aide.agent_backtrack.Agent`` and loads
``utils/config_backtrack.yaml`` — the config whose ``SearchConfig`` actually defines
``use_mcts``, ``mcts_method``, ``error_backtrack_threshold`` and the Thompson-Sampling
priors — so tree-search node selection is driven by Thompson Sampling / UCB / random.

Run it directly (any config value can be overridden on the CLI via OmegaConf):

    python -m aide.run_backtrack \
        data_dir=/path/to/comp/prepared/public \
        desc_file=/path/to/full_instructions.txt \
        exp_name=aide_mcts \
        agent.search.use_mcts=true \
        agent.search.mcts_method=thompson \
        agent.search.error_backtrack_threshold=3

Set ``agent.search.use_mcts=false`` for the control run (random node selection).
"""

import atexit
import logging
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf
from rich.status import Status

from . import backend
from .agent_backtrack import Agent
from .interpreter import Interpreter
from .journal import Journal
from .run import VerboseFilter, journal_to_string_tree
from .utils.config_backtrack import (
    load_cfg,
    load_task_desc,
    prep_agent_workspace,
    save_run,
)

logger = logging.getLogger("aide")

# Default config for the backtracking agent (defines the MCTS / Thompson knobs).
BACKTRACK_CONFIG = Path(__file__).parent / "utils" / "config_backtrack.yaml"


def run():
    cfg = load_cfg(BACKTRACK_CONFIG)

    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.log_dir / "aide.log")
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.addFilter(VerboseFilter())

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "aide.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.addFilter(VerboseFilter())

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting backtracking run "{cfg.exp_name}"')
    logger.info(
        f"[search policy] use_mcts={getattr(cfg.agent.search, 'use_mcts', False)} "
        f"method={getattr(cfg.agent.search, 'mcts_method', 'n/a')} "
        f"error_backtrack_threshold={getattr(cfg.agent.search, 'error_backtrack_threshold', 0)}"
    )

    task_desc = load_task_desc(cfg)
    task_desc_str = backend.compile_prompt_to_md(task_desc)
    logger.info(f"Task description:\n{task_desc_str}", extra={"verbose": True})

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        # only remove the workspace if the run produced nothing
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir, ignore_errors=True)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(task_desc=task_desc, cfg=cfg, journal=journal)
    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec)  # type: ignore
    )

    global_step = len(journal)

    def exec_callback(*args, **kwargs):
        return interpreter.run(*args, **kwargs)

    while global_step < cfg.agent.steps:
        agent.step(exec_callback=exec_callback)
        if global_step == cfg.agent.steps - 1:
            logger.info(journal_to_string_tree(journal))
        save_run(cfg, journal)
        global_step = len(journal)

    interpreter.cleanup_session()


if __name__ == "__main__":
    run()
