"""configuration and setup utils"""

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Hashable, cast, Literal

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from search.journal import Journal, filter_journal

from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("ml-master")

# Register OmegaConf resolver for environment variables
# This allows ${oc.env:ENV_VAR} syntax to resolve to os.getenv("ENV_VAR")
if not OmegaConf.has_resolver("env"):
    OmegaConf.register_new_resolver(
        "env",
        lambda key, default=None: os.getenv(key, default) if default is not None else os.getenv(key, ""),
        replace=False
    )


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str

@dataclass
class LinearDecay:
    alpha: float

@dataclass
class ExponentialDecay:
    gamma: float

@dataclass
class PiecewiseDecay:
    alpha: float
    phase_ratios: list

@dataclass
class DynamicPiecewiseDecay:
    alpha: float
    phase_ratios: list

@dataclass
class DecayConfig:
    decay_type: str
    exploration_constant: float
    lower_bound: float
    linear_decay: LinearDecay
    exponential_decay: ExponentialDecay
    piecewise_decay: PiecewiseDecay
    dynamic_piecewise_decay: DynamicPiecewiseDecay
    

@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    invalid_metric_upper_bound: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    max_improve_failure: int
    parallel_search_num: int

@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool
    convert_system_to_user: bool
    obfuscate: bool
    check_format: bool
    save_all_submission: bool
    steerable_reasoning: bool
    require_hyperparameter_tuning: bool
    code: StageConfig
    feedback: StageConfig
    search: SearchConfig
    decay: DecayConfig

@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool


@dataclass
class Config(Hashable):
    data_dir: Path
    dataset_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    agent: AgentConfig
    start_cpu_id: str
    cpu_number: str


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if current_index := int(p.name.split("-")[0]) > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config_mcts.yaml", use_cli_args=True
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cli_cfg = OmegaConf.from_cli()
        # Remove shell-only kwargs that aren't part of the Python Config schema
        # These are used by start.sh but not by Python code
        shell_only_kwargs = ['start_cpu', 'cpus_per_task']
        for key in shell_only_kwargs:
            if key in cli_cfg:
                del cli_cfg[key]
        cfg = OmegaConf.merge(cfg, cli_cfg)
    return cfg

def load_cfg(path: Path = Path(__file__).parent / "config_mcts.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def prep_cfg(cfg: Config):
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    # generate experiment name and prefix with consecutive index
    cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)

    cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
    cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    # Remove shell-only kwargs that aren't part of the Python Config schema
    # These are used by start.sh but not by Python code (they're already filtered in _load_cfg, but double-check here)
    shell_only_kwargs = ['start_cpu', 'cpus_per_task']
    for key in shell_only_kwargs:
        if key in cfg:
            del cfg[key]
    
    # Merge with schema
    cfg = OmegaConf.merge(cfg_schema, cfg)
    
    # Resolve environment variable interpolations manually
    # OmegaConf ${VAR} interpolation looks for config keys, not env vars
    # We need to manually resolve ${ENV_VAR} patterns to actual environment variable values
    def resolve_env_interpolation(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}") and ":" not in value:
            # Pattern: ${ENV_VAR} - resolve from environment
            env_var = value[2:-1].strip()
            env_value = os.getenv(env_var)
            if env_value is None:
                raise ValueError(f"Environment variable `{env_var}` is not set (referenced in config as `${{{env_var}}}`)")
            return env_value
        return value
    
    # Resolve api_key fields if they contain env var interpolation
    # Access values directly and replace if they're interpolation patterns
    try:
        # Get raw values without triggering interpolation
        cfg_dict = OmegaConf.to_container(cfg, resolve=False)
        
        # Helper to recursively resolve env interpolations in nested dicts
        def resolve_dict(d):
            if isinstance(d, dict):
                return {k: resolve_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [resolve_dict(item) for item in d]
            else:
                return resolve_env_interpolation(d)
        
        resolved_dict = resolve_dict(cfg_dict)
        # Recreate cfg from resolved dict
        cfg = OmegaConf.merge(cfg_schema, OmegaConf.create(resolved_dict))
    except Exception as e:
        logger.error(f"Failed to resolve API key environment variable: {e}")
        raise

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal: Journal):
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # filtered_journal = filter_journal(journal)
    # save journal
    # serialize.dump_json(journal, cfg.log_dir / "journal.json")
    # serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config_mcts.yaml")
    
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
    # # concatenate logs
    # with open(cfg.log_dir / "full_log.txt", "w") as f:
    #     f.write(
    #         concat_logs(
    #             cfg.log_dir / "ml-master.log",
    #             cfg.workspace_dir / "best_solution" / "node_id.txt",
    #             cfg.log_dir / "filtered_journal.json",
    #         )
    #     )


def concat_logs(chrono_log: Path, best_node: Path, journal: Path):
    content = (
        "The following is a concatenation of the log files produced.\n"
        "If a file is missing, it will be indicated.\n\n"
    )

    content += "---First, a chronological, high level log of the ml-master run---\n"
    content += output_file_or_placeholder(chrono_log) + "\n\n"

    content += "---Next, the ID of the best node from the run---\n"
    content += output_file_or_placeholder(best_node) + "\n\n"

    content += "---Finally, the full journal of the run---\n"
    content += output_file_or_placeholder(journal) + "\n\n"

    return content


def output_file_or_placeholder(file: Path):
    if file.exists():
        if file.suffix != ".json":
            return file.read_text()
        else:
            return json.dumps(json.loads(file.read_text()), indent=4)
    else:
        return f"File not found."