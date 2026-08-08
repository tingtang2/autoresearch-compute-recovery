import csv
import glob
import shutil
import logging
import random
import os
import tempfile
import time
from typing import Any, Callable, cast, Tuple, List, Literal
import math
import humanize
from backend import FunctionSpec, compile_prompt_to_md, query, r1_query, gpt_query
from interpreter.interpreter_parallel import ExecutionResult
from search.journal import Journal
from search.mcts_node import MCTSNode
import utils.data_preview as data_preview
from utils.config_mcts import Config
from utils.metric import MetricValue, WorstMetricValue
from utils.response import extract_code, extract_text_up_to_code, wrap_code, extract_review
from utils.server_utils import call_validate
from utils.mcts import linear_decay, exponential_decay, piecewise_decay, dynamic_piecewise_decay
import threading

from agent.bug_consultant import BugConsultant

logger = logging.getLogger("ml-master")


def _read_csv_header(path) -> list:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def _maybe_coerce_submission_header(submission_path, expected_columns: list):
    """Coerce submission header and column order to match sample_submission.csv."""
    from pathlib import Path
    submission_path = Path(submission_path)
    try:
        got = _read_csv_header(submission_path)
    except Exception:
        return None, "Failed to read header"

    if got == expected_columns:
        return None, None

    norm = lambda s: s.strip().lower()
    got_norm = [norm(c) for c in got]
    expected_norm = [norm(c) for c in expected_columns]

    if not got_norm or not expected_norm:
        return None, None

    mapping = {}
    used_actual_norm = set()
    got_norm_to_actual = {}
    for actual, actual_norm in zip(got, got_norm):
        got_norm_to_actual.setdefault(actual_norm, []).append(actual)

    for exp_col, exp_col_norm in zip(expected_columns, expected_norm):
        candidates = []
        exact = got_norm_to_actual.get(exp_col_norm, [])
        candidates.extend(exact)
        if not candidates and "_" in exp_col_norm:
            suffix = exp_col_norm.split("_")[-1]
            candidates.extend(got_norm_to_actual.get(suffix, []))
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) != 1:
            return None, None
        actual = candidates[0]
        actual_norm = norm(actual)
        if actual_norm in used_actual_norm:
            return None, None
        used_actual_norm.add(actual_norm)
        mapping[exp_col] = actual

    if "id" not in got_norm or "id" not in expected_norm:
        return None, None

    try:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False, suffix=".csv",
            dir=str(submission_path.parent),
        )
        tmp_path = Path(tmp.name)
        with open(submission_path, "r", newline="", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(tmp, fieldnames=expected_columns, extrasaction="ignore")
            writer.writeheader()
            for row in reader:
                out_row = {exp: row.get(mapping[exp], "") for exp in expected_columns}
                writer.writerow(out_row)
        tmp.close()
        return tmp_path, None
    except Exception as e:
        try:
            tmp.close()
        except Exception:
            pass
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None, f"Failed to rewrite: {e}"


def format_time(time_in_sec: int):
    return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"

ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": (
                    "true if the run failed OR if required outputs are missing/invalid/inconsistent.\n"
                    "Sanity checks for `metric`: must be finite. Treat metric equal to exactly 0 or 1 as invalid/placeholder."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "If there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) "
                    "describing the empirical findings."
                ),
            },
            "metric": {
                "type": ["number", "null"],
                "description": "If the code ran successfully, report the validation metric; else null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized; false if it should be maximized.",
            },
            "cv_folds": {
                "type": ["array", "null"],
                "items": {"type": "number"},
                "description": (
                    "Extract ALL cross-validation fold scores from the output. "
                    "Look for patterns like: 'cv_folds': [0.72, 0.73, 0.72] or "
                    "'Fold 1: 0.72', 'Fold 2: 0.73', etc. "
                    "Return null if no CV fold scores are found."
                ),
            },
        },
        "required": [
            "is_bug",
            "summary",
            "metric",
            "lower_is_better",
            "cv_folds",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)


class MCTSAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        self.journal = journal
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: MCTSNode | None = None
        self.all_root = True
        self.virtual_root = MCTSNode(parent=None, plan="virtual plan", code="# virtual code", metric=WorstMetricValue(), stage="root")
        self.current_node_list = []
        self.journal.append(self.virtual_root)
        self.best_metric: float = None
        self.best_node: MCTSNode = None
        self.metric_direction: bool | None = None  # locked after first valid node
        # Try to extract competition metric from description.md (actual comp description)
        desc_md_path = cfg.data_dir / "description.md" if hasattr(cfg, 'data_dir') and cfg.data_dir else None
        desc_text = ""
        if desc_md_path and desc_md_path.exists():
            try:
                desc_text = desc_md_path.read_text()
            except Exception:
                pass
        # Fall back to task_desc if description.md not available
        self.competition_metric: str | None = self._extract_metric_name(desc_text) or self._extract_metric_name(task_desc)
        if self.competition_metric:
            logger.info(f"Extracted competition metric: {self.competition_metric}")
        self.search_start_time = None
        self.journal_lock = threading.Lock()
        self.save_node_lock = threading.Lock()
        self.start_time = time.time()

        # Debug consultant integration
        self.bug_consultant: BugConsultant | None = None
        if getattr(self.scfg, "use_bug_consultant", False):
            save_dir = (cfg.log_dir / "bug_consultant").resolve()
            self.bug_consultant = BugConsultant(
                model=self.acfg.feedback.model,
                temperature=0.3,
                save_dir=save_dir,
                cfg=cfg,
            )

    @staticmethod
    def _extract_metric_name(task_desc: str) -> str | None:
        """Extract metric name from competition description."""
        import re
        boilerplate = ['metric in the competition', 'competition-specific', 'described in']

        # Priority 1: bold metric name **metric name** (most descriptions use this)
        for pat in [
            r'evaluated (?:using|on)\s+(?:the\s+)?\*\*(.+?)\*\*',
            r'scored (?:based )?on\s+(?:the\s+)?\*\*(.+?)\*\*',
            r'metric\s+(?:is|:)\s+(?:the\s+)?\*\*(.+?)\*\*',
        ]:
            m = re.search(pat, task_desc, re.IGNORECASE)
            if m:
                result = m.group(1).strip()
                if not any(bp in result.lower() for bp in boilerplate):
                    return result

        # Priority 2: plain text metric name (letters/spaces only, stops at punctuation)
        for pat in [
            r'evaluated (?:using|on)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]+?)[\.,$\n]',
            r'scored (?:based )?on\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]+?)[\.,$\n]',
            r'metric\s+(?:is|:)\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]+?)[\.,$\n]',
            r'graded (?:based )?on\s+(?:the\s+)?([a-zA-Z][a-zA-Z\s]+?)[\.,$\n]',
        ]:
            m = re.search(pat, task_desc, re.IGNORECASE)
            if m:
                result = m.group(1).strip()
                if not any(bp in result.lower() for bp in boilerplate):
                    return result
        return None

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "transformers",
            "nltk",
            "spacy",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt
    
    @property
    def _prompt_impl_guideline(self):
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "The code must not only implement the proposed solution but also **print the evaluation metric computed on a hold-out validation set**. **Without this metric, the solution cannot be evaluated, rendering the entire code invalid.**,",
            "**Save predictions on the provided unlabeled test data to `./submission/submission.csv`.**",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            'You can use the "./working" directory to store any temporary files that your code needs to create.',
            "If you use `DataLoader`, you need to increase the parameter `num_workers` to speed up the training process.",
            "DO NOT use GridSearchCV or RandomizedSearchCV; use simple default hyperparameters. Do NOT do hyperparameter search of any kind (no Optuna, no BayesSearchCV, no manual grid loops).",
            "**Early Stopping**: For LightGBM, use `callbacks=[lgb.early_stopping(stopping_rounds=N)]`. For XGBoost, use native API: `xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)`.",
            "Compute safety: avoid unconstrained parallelism. Do NOT use `n_jobs=-1` anywhere. If you set `n_jobs` / `thread_count` / `num_threads`, cap them to `int(os.getenv('MLMASTER_NUM_THREADS', '4'))` (use the same cap consistently).",
            "**CRITICAL - Data Leakage Prevention**: No sample's label may influence both training and evaluation. This means: (1) NEVER use the target variable to create features. (2) NEVER evaluate, calibrate, or tune on any data the model was trained on. If you retrain on all data, you have no valid holdout — use only cross-validation out-of-fold predictions for any post-training decisions.",
            "At the end of the script, print a single-line JSON with prefix `MLMASTER_METRICS_JSON=` so the harness can parse metrics. Include: `valid`, `lower_is_better`, and (for CV) `cv_mean`, `cv_std`, `cv_folds` (list of fold scores). Example: `print('MLMASTER_METRICS_JSON=' + json.dumps({'valid': 0.85, 'lower_is_better': False, 'cv_mean': 0.84, 'cv_std': 0.02, 'cv_folds': [0.83, 0.85, 0.84]}))`",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"**MANDATORY - Cross-Validation**: Use {self.acfg.k_fold_validation}-fold cross-validation for evaluation. Report ALL fold scores in cv_folds (not just mean/std). The cv_folds list is required for robust model selection."
            )

        if self.competition_metric:
            impl_guideline.append(
                f"**Competition Metric**: The competition is evaluated using {self.competition_metric}. You MUST use this as your validation metric, not a proxy metric like RMSE or logloss."
            )

        return {"Implementation guideline": impl_guideline}
    
    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }
    
    def _draft(self) -> MCTSNode:
        logger.info("Starting Drafting a new Node.")
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "In order to win this competition, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "In order to complete this task, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": self.virtual_root.fetch_child_memory(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.\n",
                "- When proposing the design, take the Memory section into account.\n",
                "- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.\n",
                "- Don't propose the same modelling solution but keep the evaluation the same.\n",
                "- The solution sketch should be 3-5 sentences.\n",
                "- Propose an evaluation metric that is reasonable for this task.\n",
                "- Don't suggest to do EDA.\n",
                "- The data is already prepared and available in the `./input` directory. There is no need to unzip any files.\n",
                "- **CRITICAL - Data Leakage Prevention**: No sample's label may influence both training and evaluation. Do not use the target to create features, and do not evaluate or tune on data the model was trained on.\n",
                "- Do not subsample the dataset in draft runs; always use the full training data.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        # Bug prevention guidance as top-level prompt key
        if self.bug_consultant:
            exec_summary = self.bug_consultant.get_prevention_guidance(journal=self.journal)
            if exec_summary:
                prompt["Bug Prevention Alert"] = exec_summary

        instructions = f"\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        # Append bug prevention guidance to instructions so all model branches receive it
        if "Bug Prevention Alert" in prompt:
            instructions += f"\n\n# Bug Prevention Alert\n{prompt['Bug Prevention Alert']}\n"

        if "qwen3" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to solve task is provided below:\n {prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"<|im_start|>system\n{introduction}<|im_end|>\n<|im_start|>user{user_prompt}<|im_end|><|im_start|>assistant\n<think>Okay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}"
        elif "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to solve task is provided below:\n{prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"<｜begin▁of▁sentence｜>\n{introduction}\n<｜User｜>{user_prompt}<｜Assistant｜><think>\nOkay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}"
        elif "gpt-5" in self.acfg.code.model or self.acfg.steerable_reasoning == False:
            user_prompt = f"""
# Task description
{prompt['Task description']}

# Memory
The memory of previous solutions used to solve task is provided below:
{prompt['Memory']}

{instructions}

# Data preview
{self.data_preview}
"""
            prompt_complete = [
                    {"role": "system", "content": prompt['Introduction']},
                    {"role": "user", "content": user_prompt}
            ]
        self.virtual_root.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=self.virtual_root, stage="draft", local_best_node=self.virtual_root)
        logger.info(f"Drafted a new node {new_node.id} successfully!")
        return new_node

    def _improve(self, parent_node: MCTSNode) -> MCTSNode:
        logger.info(f"Starting Improving Node {parent_node.id}.")
        introduction = (
            "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            )
        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Memory": parent_node.fetch_child_memory(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        # Bug prevention guidance as top-level prompt key
        if self.bug_consultant:
            exec_summary = self.bug_consultant.get_prevention_guidance(journal=self.journal)
            if exec_summary:
                prompt["Bug Prevention Alert"] = exec_summary

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "- The solution sketch should be a brief natural language description of how the previous solution can be improved.\n",
                "- You should be very specific and should only propose a single actionable improvement.\n",
                "- This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.\n",
                "- When proposing the design, take the Memory section into account.\n",
                "- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.\n",
                "- The solution sketch should be 3-5 sentences.\n",
                "- If there is any debug subsampling code in the previous solution (e.g., data sampling, `frac=...`, `nrows=...`, `head(...)`, `# DEBUG`), remove it and use the full dataset.\n",
                "- NEVER add target encoding or any feature engineering that involves the target variable - this causes data leakage. Do not evaluate or tune on data the model was trained on.\n",
                "- Don't suggest to do EDA.\n",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        output = wrap_code(parent_node.term_out, lang="")

        instructions = "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        # Append bug prevention guidance to instructions so all model branches receive it
        if "Bug Prevention Alert" in prompt:
            instructions += f"\n\n# Bug Prevention Alert\n{prompt['Bug Prevention Alert']}\n"

        if "qwen3" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            qwen3_user_prompt = f"\n# Task description\n{prompt['Task description']}\n# Memory\nThe memory of previous solutions used to improve performance is provided below:\n {prompt['Memory']}\n{instructions}"
            prompt_complete = f"<|im_start|>system\n{introduction}<|im_end|>\n<|im_start|>user{qwen3_user_prompt}<|im_end|><|im_start|>assistant\n<think>Okay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made attempts with the following code:\n{prompt['Previous solution']['Code']}\nThe execution of this code yielded the following results:\n{output}\nI believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
        elif "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n\n# Memory\nThe memory of previous solutions used to improve performance is provided below:\n {prompt['Memory']}\n\n{instructions}"
            prompt_complete = f"<｜begin▁of▁sentence｜>{introduction}<｜User｜>{user_prompt}<｜Assistant｜><think>\nOkay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made attempts with the following code:\n{prompt['Previous solution']['Code']}\nThe execution of this code yielded the following results:\n{output}\nI believe that there is likely still room for optimization based on this code, and perhaps some aspects could be further refined and improved to enhance its performance."
        elif "gpt-5" in self.acfg.code.model or self.acfg.steerable_reasoning == False:
            user_prompt = f"""
# Task description
{prompt['Task description']}
# Memory
The memory of previous solutions used to improve performance is provided below:
{prompt['Memory']}

{instructions}

# Data preview
{self.data_preview}

# Previous solution
{prompt['Previous solution']['Code']}

# Execution output
{output}
"""
            prompt_complete = [
                    {"role": "system", "content": prompt['Introduction']},
                    {"role": "user", "content": user_prompt}
            ]
        parent_node.add_expected_child_count()

        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=parent_node, stage="improve", local_best_node=parent_node.local_best_node)
        logger.info(f"Improving node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _get_original_plan(self, node: "MCTSNode") -> str:
        """Walk up the tree to find the original (non-debug) plan."""
        current = node
        while current.parent and getattr(current, "stage_name", None) == "debug":
            current = current.parent
        return current.plan or ""

    def _debug(self, parent_node: MCTSNode) -> MCTSNode:
        logger.info(f"Starting Debugging Node {parent_node.id}.")
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, "
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        if self.acfg.check_format:
            introduction = (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug and/or did not produce a submission.csv, or the generated submission.csv was in an incorrect format,"
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )

        prompt: Any = {
            "Introduction": introduction,
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "- BLOCKLIST CHECK: Read 'Historical Bug Context' and 'Current Bug Trial History' — those approaches have ALREADY FAILED and WILL CRASH AGAIN if you use them. You MUST use a DIFFERENT approach.\n",
                "- You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.\n",
                "- **FOR QUICK DEBUGGING** (ONLY for datasets >50,000 rows): You may subsample to 10%, but MUST check class counts first to avoid crashes:\n```python\nmin_class_count = y.value_counts().min()\nn_splits = 5\nif min_class_count >= n_splits:\n    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)\nelse:\n    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # FALLBACK\n```\nNEVER use StratifiedKFold on subsampled data without checking class counts! Add `# DEBUG: Using 10% subsample` comment so it can be removed once fixed.\n",
                "- Don't suggest to do EDA.\n",
            ],
        }
        prompt["Instructions"] |= {
            "Plan adherence": [
                "Honor the IDEA of the Approved Solution Plan (model choice, CV strategy, feature approach)",
                "Implementation DETAILS (API calls, parameter names) CAN change to avoid bugs",
                "Example: Plan says 'early stopping' → choose a working implementation (e.g., remove the feature or use alternative method)",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        # Historical bug context for debug
        debug_history_section = ""
        active_trials_section = ""
        if self.bug_consultant:
            retrieval = self.bug_consultant.retrieve_relevant_context(
                current_error_type=parent_node.exc_type or "Unknown",
                current_error_msg=parent_node.term_out or "",
                current_code=parent_node.code,
                original_plan=self._get_original_plan(parent_node),
            )
            debug_history_section = self.bug_consultant.format_context_for_actor(retrieval)
            try:
                active_trials_section = self.bug_consultant.format_active_trial_history(
                    f"bug_{parent_node.step}", max_trials=5
                )
            except Exception:
                active_trials_section = ""

        # Build prompt: Bug context at TOP (before instructions) for maximum visibility
        bug_context_block = ""
        if debug_history_section:
            bug_context_block += f"\n# Historical Bug Context (Curated)\n{debug_history_section}\n"
        if active_trials_section:
            bug_context_block += f"\n# Current Bug Trial History\n{active_trials_section}\n"

        instructions = ""
        if bug_context_block:
            instructions += bug_context_block
        instructions += "\n# Instructions\n\n"
        instructions += compile_prompt_to_md(prompt["Instructions"], 2)

        if "qwen3" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            qwen3_user_prompt = f"\n# Task description\n{prompt['Task description']}\n{instructions}"
            prompt_complete = f"<|im_start|>system\n{introduction}<|im_end|>\n<|im_start|>user{qwen3_user_prompt}<|im_end|><|im_start|>assistant\n<think>Okay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made an attempt with the following code:\n{prompt['Previous (buggy) implementation']}\nHowever, there are the following issues with this code:\n{prompt['Execution output']}\nI hold the view that the underlying reasons giving rise to the emergence of this issue are:\n{parent_node.analysis}\nThe previous solution had a bug and/or did not produce a submission.csv. I will try to fix the bug."
        elif "deepseek" in self.acfg.code.model and self.acfg.steerable_reasoning== True:
            user_prompt = f"\n# Task description\n{prompt['Task description']}\n{instructions}"
            prompt_complete = f"<｜begin▁of▁sentence｜>{prompt['Introduction']}<｜User｜>{user_prompt}<｜Assistant｜><think>\nOkay! Now, I will focus my efforts on successfully completing this current task.\nBefore completing this task, first of all, I need to analyze and understand the relevant dataset. The information of the dataset is as follows: \n{self.data_preview}\nRegarding this task, I previously made an attempt with the following code:\n{prompt['Previous (buggy) implementation']}\nHowever, there are the following issues with this code:\n{prompt['Execution output']}\nI hold the view that the underlying reasons giving rise to the emergence of this issue are:\n{parent_node.analysis}\nThe previous solution had a bug and/or did not produce a submission.csv, or the generated submission.csv was in an incorrect format.I will try to fix the bug."
        elif "gpt-5" in self.acfg.code.model or self.acfg.steerable_reasoning == False:
            user_prompt = f"""
# Task description
{prompt['Task description']}

{instructions}

# Data preview
{self.data_preview}

# Previous (buggy) implementation
{prompt['Previous (buggy) implementation']}

# Execution output
{prompt['Execution output']}
"""
            prompt_complete = [
                    {"role": "system", "content": prompt['Introduction']},
                    {"role": "user", "content": user_prompt}
            ]        

        parent_node.add_expected_child_count()
        plan, code = self.plan_and_code_query(prompt_complete)
        new_node = MCTSNode(plan=plan, code=code, parent=parent_node, stage="debug", local_best_node=parent_node.local_best_node)
        logger.info(f"Debugging node {parent_node.id} to create new node {new_node.id}")
        return new_node
    
    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            if "gpt-5" in self.acfg.code.model:
                completion_text = gpt_query(
                    prompt = prompt,
                    temperature=self.acfg.code.temp,
                    model=self.acfg.code.model,
                    cfg=self.cfg
                )
            else:
                completion_text = r1_query(
                    prompt = prompt,
                    temperature=self.acfg.code.temp,
                    model=self.acfg.code.model,
                    cfg=self.cfg
                )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore
    
    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def backpropagate(self, node: MCTSNode, value: float, add_to_tree=True):
        logger.info(f"node {node.id} start backpropagating with reward {value}.")
        while node != None:
            if node.is_buggy is False and node.parent.is_buggy is True:
                node.parent.is_debug_success = True
            elif node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
                node.parent.is_debug_success = True
            if node.parent and node.parent.stage != "root":
                node.parent.continue_improve = node.continue_improve
            if node.stage == "draft" and node.lock:
                node.lock = False
                logger.info(f"Draft node {node.id} is unlocked.")
            if node.improve_failure_depth>0:
                node.improve_failure_depth = 0
            node.update(value, add_to_tree)
            node = node.parent
            
    @staticmethod
    def _parse_metrics_json(term_out: str) -> dict | None:
        """Parse MLMASTER_METRICS_JSON=<json> (or AIDE_METRICS_JSON=) from execution output."""
        import json as _json
        for line in term_out.splitlines():
            line = line.strip()
            for prefix in ("MLMASTER_METRICS_JSON=", "MLMASTER_METRICS=", "AIDE_METRICS_JSON=", "AIDE_METRICS="):
                if line.startswith(prefix):
                    payload = line[len(prefix):].strip()
                    if not payload:
                        return None
                    try:
                        obj = _json.loads(payload)
                        return obj if isinstance(obj, dict) else None
                    except _json.JSONDecodeError:
                        return None
        return None

    @staticmethod
    def _validate_cv_folds(folds: list[float]) -> tuple[bool, str | None]:
        """Check if CV folds are valid (not placeholder values)."""
        if not folds:
            return False, None
        if all(f == 0.0 for f in folds):
            return False, f"Invalid CV folds: all scores are 0.0 {folds}. Model failed to make meaningful predictions."
        if all(f == 1.0 for f in folds):
            return False, f"Invalid CV folds: all scores are 1.0 {folds}. Likely data leakage or broken evaluation."
        if len(folds) > 1 and len(set(folds)) == 1:
            return False, f"Suspicious CV folds: all {len(folds)} folds have identical score {folds[0]}. Real CV should show variance."
        return True, None

    def _apply_metrics_and_cv_validation(self, node: MCTSNode, response: dict) -> None:
        """Apply structured metrics parsing and CV fold validation to a node.

        Two-layer approach:
        1. Try JSON parsing from MLMASTER_METRICS_JSON (structured, reliable)
        2. Fall back to LLM-extracted cv_folds from reviewer response

        Modifies response dict in-place and sets node attributes.
        """
        term_out = node.term_out or ""

        # Layer 1: Try structured JSON metrics
        parsed = self._parse_metrics_json(term_out)
        if parsed is not None:
            # Prefer structured valid metric over LLM extraction
            valid = parsed.get("valid") or parsed.get("val") or parsed.get("metric")
            if isinstance(valid, (int, float)) and not isinstance(valid, bool):
                response["metric"] = float(valid)
            lib = parsed.get("lower_is_better")
            if isinstance(lib, bool):
                response["lower_is_better"] = lib

        # Coerce metric
        metric = response.get("metric")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            response["metric"] = None
        else:
            metric_f = float(metric)
            response["metric"] = metric_f if math.isfinite(metric_f) else None

        # Hard-coded guardrail: treat exact 0/1 metrics as invalid placeholders (matches AIDE)
        if response.get("metric") in (0, 1, 0.0, 1.0):
            response["metric"] = None
            response["is_bug"] = True

        # --- CV fold validation ---
        cv_folds = None
        cv_validation_error = None

        # Layer 1: JSON-parsed cv_folds
        if parsed is not None:
            raw_folds = parsed.get("cv_folds") or parsed.get("folds")
            if isinstance(raw_folds, list) and len(raw_folds) > 0:
                try:
                    candidate = [float(v) for v in raw_folds]
                    is_valid, err = self._validate_cv_folds(candidate)
                    if is_valid:
                        cv_folds = candidate
                    else:
                        cv_validation_error = err
                except (ValueError, TypeError):
                    pass

        # Layer 2: LLM-extracted cv_folds from reviewer
        if cv_folds is None and cv_validation_error is None:
            raw_folds = response.get("cv_folds")
            if isinstance(raw_folds, list) and len(raw_folds) > 0:
                try:
                    candidate = [float(v) for v in raw_folds]
                    is_valid, err = self._validate_cv_folds(candidate)
                    if is_valid:
                        cv_folds = candidate
                    else:
                        cv_validation_error = err
                except (ValueError, TypeError):
                    pass

        # Store cv data on node (if node supports these attributes)
        if cv_folds:
            import numpy as np
            node.cv_folds = cv_folds
            node.cv_mean = float(np.mean(cv_folds))
            node.cv_std = float(np.std(cv_folds))
            logger.info(f"CV metrics: mean={node.cv_mean:.6f}, std={node.cv_std:.6f}, folds={len(cv_folds)}")

        # If cv_folds validation failed, mark as buggy
        if cv_validation_error is not None:
            response["is_bug"] = True
            response["metric"] = None
            node.analysis = f"[CV VALIDATION ERROR] {cv_validation_error}\n\nOriginal: {node.analysis or ''}"
            logger.error(f"Node {node.id} marked buggy: {cv_validation_error}")

    def parse_exec_result(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)

            introduction = (
                "You are a Kaggle grandmaster attending a competition. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
            if self.acfg.obfuscate:
                introduction = (
                    "You are an expert machine learning engineer attempting a task. "
                    "You have written code to solve this task and now need to evaluate the output of the code execution. "
                    "You should determine if there were any bugs as well as report the empirical findings."
                )
            review_task_desc = self.task_desc
            if self.competition_metric:
                review_task_desc += f"\n\n**Competition Metric**: The competition is evaluated using {self.competition_metric}. You MUST use this as the validation metric when reporting results."
            prompt = {
                "Introduction": introduction,
                "Task description": review_task_desc,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }

            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=review_func_spec,
                    model=self.acfg.feedback.model,
                    temperature=self.acfg.feedback.temp,
                    convert_system_to_user=self.acfg.convert_system_to_user,
                    cfg=self.cfg
                ),
            )

            node.analysis = response.get("summary", "")

            # Apply structured metrics parsing, 0/1 guardrail, and CV validation
            self._apply_metrics_and_cv_validation(node, response)

            # do an extra check, to catch cases where judge fails
            has_csv_submission = (
                self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
            ).exists()

            # Override false-positive: replace_submission_name() renames submission.csv
            # to submission_<hash>.csv for parallel safety, but reviewer sees the hash
            # filename and incorrectly marks is_bug=True. Trust disk evidence.
            # Do NOT override if metric is exactly 0 or 1 (likely invalid/placeholder).
            metric_val = response.get("metric")
            metric_suspicious = metric_val is not None and metric_val in (0, 1, 0.0, 1.0)
            if response["is_bug"] and has_csv_submission and metric_val is not None and not metric_suspicious and node.exc_type is None:
                logger.info(f"Overriding reviewer is_bug=True for node {node.id}: submission exists on disk with metric={metric_val}")
                response["is_bug"] = False

            if response["is_bug"] or node.exc_type is not None or response["metric"] is None or not has_csv_submission:
                if response["is_bug"]:
                    logger.warning(f"Node {node.id} is marked as buggy because the response['is_bug'] is True.")
                elif node.exc_type is not None:
                    logger.warning(f"Node {node.id} is marked as buggy because the node.exc_type is not None.")
                elif response["metric"] is None:
                    logger.warning(f"Node {node.id} is marked as buggy because response['metric'] is None.")
                else:
                    logger.warning(f"Node {node.id} is marked as buggy because has_csv_submission is False.")

            node.is_buggy = (
                response["is_bug"]
                or node.exc_type is not None
                or response["metric"] is None
                or not has_csv_submission
            )
            if not node.is_buggy and self.acfg.check_format:
                exp_id = self.cfg.exp_name.split("_")[0]
                logger.info(f"Start checking the format of submission.csv of node {node.id}")
                status, res = call_validate(exp_id=exp_id, submission_path=self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv")
                if status:
                    if not res['is_valid']:
                        logger.warning(f"Node {node.id} is marked as buggy because file: submission.csv is invalid.")
                        node.is_valid = False
                        node._term_out.append(f"\n{res['result']}")
                        node.analysis = "This previous solution runs without any bugs, but the format of the generated submission file is incorrect."
                    else:
                        node.is_valid = True
                        logger.info(f"Node {node.id} file: submission.csv is valid.")
                else:
                    logger.error(f"An unexpected error occurred: {res}, skip this stage.")
                    node.is_valid = True

            if node.is_buggy:
                logger.info(
                    f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
                )
                node.metric = WorstMetricValue()
            else:
                logger.info(f"Parsed results: Node {node.id} is not buggy")
                node.metric = MetricValue(
                    response["metric"], maximize=not response["lower_is_better"]
                )
                # Lock metric direction from first valid node
                if self.metric_direction is None:
                    self.metric_direction = node.metric.maximize
                    logger.info(f"Locked metric direction: maximize={self.metric_direction}")
                else:
                    node.metric.maximize = self.metric_direction
            return node
        except Exception as e:
            logger.warning(f"parse result with tool error:{e}")
            logger.info("parse_exec_result_without_tool")
            return self.parse_exec_result_without_tool(node, exec_result)

    def parse_exec_result_without_tool(self, node: MCTSNode, exec_result: ExecutionResult) -> MCTSNode:
        logger.info(f"Agent is parsing execution results for node {node.id} without using tool.")
        node.absorb_exec_result(exec_result)
        introduction = (
            "You are a Kaggle grandmaster attending a competition. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings.\n\n"
            "You should evaluate the output of the code in Implementation. The review must be submitted in a specific JSON format with the following fields:\n\n"
            "- is_bug (boolean): true if the run failed OR if required outputs are missing/invalid/inconsistent. Sanity checks for metric: must be finite. Treat metric equal to exactly 0 or 1 as invalid/placeholder.\n"
            "- summary (string): If there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.\n"
            "- metric (number): If the code ran successfully, report the value of the validation metric here. If the code failed, this field should be set to null.\n"
            "- lower_is_better (boolean): This field indicates whether the metric should be minimized. If a lower value of the metric represents better performance (e.g., for Mean Squared Error), set this to true. If a higher value represents better performance (e.g., for accuracy), set this to false.\n"
            "- cv_folds (array or null): Extract ALL cross-validation fold scores from the output. Look for patterns like 'cv_folds': [0.72, 0.73, 0.72] or 'Fold 1: 0.72', etc. Return null if no CV fold scores are found.\n\n"
            """The review must be submitted in the following JSON format in a single markdown code block (wrapped in ```):
```json
{
    "is_bug": true,
    "summary": "The code encountered an error during execution. The CSV file was not generated.",
    "metric": null,
    "lower_is_better": true,
    "cv_folds": null
}
```
"""
            ""
        )
        if self.acfg.obfuscate:
            introduction = (
                "You are an expert machine learning engineer attempting a task. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            )
        review_task_desc = self.task_desc
        if self.competition_metric:
            review_task_desc += f"\n\n**Competition Metric**: The competition is evaluated using {self.competition_metric}. You MUST use this as the validation metric when reporting results."
        prompt = {
            "Introduction": introduction,
            "Task description": review_task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }
        try:
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
                cfg=self.cfg
            )
        except Exception as e:
            logger.info("parse without tool fail, try one more time.")
            completion_text = r1_query(
                prompt=prompt,
                temperature=self.acfg.code.temp,
                cfg=self.cfg
            )
        response = cast(
            dict,
            extract_review(completion_text)
        )

        node.analysis = response.get("summary", "")

        # Apply structured metrics parsing, 0/1 guardrail, and CV validation
        self._apply_metrics_and_cv_validation(node, response)

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"
        ).exists()

        # Override false-positive: replace_submission_name() renames submission.csv
        # to submission_<hash>.csv for parallel safety, but reviewer sees the hash
        # filename and incorrectly marks is_bug=True. Trust disk evidence.
        # Do NOT override if metric is exactly 0 or 1 (likely invalid/placeholder).
        metric_val = response.get("metric")
        metric_suspicious = metric_val is not None and metric_val in (0, 1, 0.0, 1.0)
        if response["is_bug"] and has_csv_submission and metric_val is not None and not metric_suspicious and node.exc_type is None:
            logger.info(f"Overriding reviewer is_bug=True for node {node.id}: submission exists on disk with metric={metric_val}")
            response["is_bug"] = False

        if response["is_bug"] or node.exc_type is not None or response["metric"] is None or not has_csv_submission:
            if response["is_bug"]:
                logger.warning(f"Node {node.id} is marked as buggy because the response['is_bug'] is True.")
            elif node.exc_type is not None:
                logger.warning(f"Node {node.id} is marked as buggy because the node.exc_type is not None.")
            elif response["metric"] is None:
                logger.warning(f"Node {node.id} is marked as buggy because response['metric'] is None.")
            else:
                logger.warning(f"Node {node.id} is marked as buggy because has_csv_submission is False.")

        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
            or not has_csv_submission
        )
        if not node.is_buggy and self.acfg.check_format:
            exp_id = self.cfg.exp_name.split("_")[0]
            logger.info(f"Start checking the format of submission.csv of node {node.id}")
            status, res = call_validate(exp_id=exp_id, submission_path=self.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv")
            if status:
                if not res['is_valid']:
                    logger.warning(f"Node {node.id} is marked as buggy because file: submission.csv is invalid.")
                    node.is_valid = False
                    node._term_out.append(f"\n{res['result']}")
                    node.analysis = "This previous solution runs without any bugs, but the format of the generated submission file is incorrect."
                else:
                    node.is_valid = True
                    logger.info(f"Node {node.id} file: submission.csv is valid.")
            else:
                logger.error(f"An unexpected error occurred: {res}, skip this stage.")

        if node.is_buggy:
            logger.info(
                f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
            )
            node.metric = WorstMetricValue()
        else:
            logger.info(f"Parsed results: Node {node.id} is not buggy")
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
            # Lock metric direction from first valid node
            if self.metric_direction is None:
                self.metric_direction = node.metric.maximize
                logger.info(f"Locked metric direction: maximize={self.metric_direction}")
            else:
                node.metric.maximize = self.metric_direction
        return node

    def select(self, node: MCTSNode):
        logger.info(f"[select] Processing node: {node.id}")
        stall_count = 0
        while node and not node.is_terminal:
            # Check global shutdown signal
            from main_mcts import _shutdown_event
            if _shutdown_event.is_set():
                logger.info("[select] Shutdown signal received, returning current node.")
                return node

            prev_node = node
            if not node.is_fully_expanded_with_expected(scfg=self.scfg):
                if node.is_buggy and node.is_debug_success is True:
                    node = self.uct_select(node)
                elif node.continue_improve and len(node.children)>0:
                    node = self.uct_select(node)
                else:
                    logger.info(f"Node {node.id} is not fully expanded, expanding")
                    return node
            else:
                node = self.uct_select(node)
            # Detect stall: selection returned the same node (all children locked)
            if node is prev_node and self.is_root(node):
                stall_count += 1
                if stall_count >= 3:
                    logger.warning(f"[select] All children of {node.id} locked, waiting for unlock...")
                    time.sleep(10)
                    stall_count = 0
            else:
                stall_count = 0
        logger.info(f"[select]choose a node for expanding: {node.id}")
        return node

    def get_C(self):
        dcfg =  self.cfg.agent.decay
        if dcfg.decay_type == "linear":
            linear_cfg = dcfg.linear_decay
            return linear_decay(
                t=self.current_step, 
                initial_C=dcfg.exploration_constant,
                lower_bound=dcfg.lower_bound,
                alpha=linear_cfg.alpha
            )
        
        elif dcfg.decay_type == "exponential":
            exponential_cfg = dcfg.exponential_decay
            return exponential_decay(
                t=self.current_step,
                initial_C=self.scfg.exploration_constant,
                lower_bound=dcfg.lower_bound,
                gamma=exponential_cfg.gamma,
            )
        
        elif dcfg.decay_type == "piecewise":
            piecewise_cfg = dcfg.piecewise_decay
            n1 = self.scfg.num_drafts*(self.scfg.num_improves ** 2)
            n2 = round(self.acfg.steps*piecewise_cfg.phase_ratios[0])
            t1 = min(n1,n2)
            t2 = round(self.acfg.steps*piecewise_cfg.phase_ratios[1])
            return piecewise_decay(
                t=self.current_step, 
                initial_C=dcfg.exploration_constant,
                T1=t1,
                T2=t2,
                lower_bound=dcfg.lower_bound
            )
        
        elif dcfg.decay_type == "dynamic_piecewise":
            dynamic_piecewise_cfg = dcfg.dynamic_piecewise_decay
            logger.info(f"dynamic_piecewise_cfg.phase_ratios = {dynamic_piecewise_cfg.phase_ratios}")
            return dynamic_piecewise_decay(
                steps_limit=self.acfg.steps,
                n_nodes=self.current_step,
                initial_C=dcfg.exploration_constant,
                start_time=self.search_start_time,
                time_limit=self.acfg.time_limit,
                alpha=dynamic_piecewise_cfg.alpha,
                lower_bound=dcfg.lower_bound,
                phase_ratios=dynamic_piecewise_cfg.phase_ratios
            )
        else:
            return dcfg.exploration_constant

    def uct_select(self, node: MCTSNode):
        if self.is_root(node):
            filtered_children = [child for child in node.children if not child.lock]
            locked_count = len(node.children) - len(filtered_children)
            if locked_count > 0:
                logger.info(f"For node {node.id}, there are {locked_count}/{len(node.children)} is locked.")
            selected_node = node
            if len(filtered_children) > 0:
                selected_node = max(filtered_children, key=lambda child: child.uct_value(exploration_constant = self.get_C()))
            elif len(node.children) > 0:
                # All children locked — sleep to avoid spin loop
                from main_mcts import _shutdown_event
                if _shutdown_event.is_set():
                    return node
                time.sleep(5)

            if selected_node.stage == "draft":
                selected_node.lock = True
                logger.info(f"Draft node {selected_node.id} is locked.")
            return selected_node
        else:
            return max(node.children, key=lambda child: child.uct_value(exploration_constant = self.get_C()))

    
    def check_improvement(self, cur_node: MCTSNode, parent_node: MCTSNode):
        improvement = 0
        should_backpropagate = False
        local_best_node = cur_node.local_best_node
        local_best_metric = local_best_node.metric.value
        if cur_node.is_buggy is False:
            new_metric = cur_node.metric.value
            if parent_node.is_buggy:
                logger.info(f"Successfully Debug the error in node {parent_node.id}. Recovered node {cur_node.id} will continue being explored.")
                # Don't force backpropagate — let recovered node flow into metric comparison
                # so it can be improved further (allows further improvement)
            if new_metric and local_best_metric:
                improvement = new_metric - local_best_metric if cur_node.metric.maximize else local_best_metric - new_metric
                if improvement < self.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < self.scfg.max_improve_failure:
                    local_best_node.improve_failure_depth += 1
                    logger.warning(f"Compared to Node {local_best_node.id}, Node {cur_node.id} metric improvement ({improvement}) below threshold ({self.scfg.metric_improvement_threshold}), try one more time({local_best_node.improve_failure_depth}/{self.scfg.max_improve_failure})")
                    cur_node.continue_improve = True
                elif improvement < self.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth >= self.scfg.max_improve_failure:
                    logging.warning(f"The number of improvement attempts for the local best node has reached its maximum limit {self.scfg.max_improve_failure}.")
                    cur_node.continue_improve = False
                    should_backpropagate = True
                    cur_node.is_terminal = True
                else:
                    logger.info(f"Compared to Node {local_best_node.id}, Node {cur_node.id} metric improvement ({improvement}) above threshold ({self.scfg.metric_improvement_threshold}), continue improving.")
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
            elif new_metric:
                logger.info(f"No local best node was found among the previous nodes; the current node {cur_node.id} is assigned as the local best")
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
            else:
                logger.warning(f"No local best node was found among the previous nodes; The current node {cur_node.id} has no errors, but contains an empty metric value.")
                should_backpropagate = True
        elif cur_node.is_buggy is None:
            logger.warning(f"Node {cur_node.id} is_buggy is None!")
            should_backpropagate = True
        else:
            if cur_node.debug_depth >= self.scfg.back_debug_depth:
                should_backpropagate = True
                if cur_node.debug_depth >= self.scfg.max_debug_depth:
                    cur_node.is_terminal = True

        if should_backpropagate:
            reward = self.get_node_reward(cur_node)
            self.backpropagate(cur_node, reward)
        else:
            self.current_node_list.append(cur_node)
        return should_backpropagate
    
    def get_node_reward(self, node: MCTSNode):
        reward = 0
        if node.is_buggy is True or node.is_buggy is None:
            reward = -1
        elif node.is_buggy is False and node.metric.value is None:
            reward = -1
        else:
            if node.metric.value and self.best_metric:
                improvement = node.metric.value - self.best_metric if node.metric.maximize else self.best_metric - node.metric.value
                if improvement > 0:
                    logger.info(f"Node {node.id} is better than the best node {self.best_node.id} now!")
                    reward += 1
            if node.parent.is_buggy is True:
                reward += 1
            else:
                reward += 1
        return reward
            
    def is_root(self, node: MCTSNode):
        return node.id is self.virtual_root.id
    
    def check_metric_valid(self, node: MCTSNode, upper_bound=50):
        '''If the metric values between nodes differ by an upper bound multiple, it is highly likely that there is an invalid metric'''
        upper_bound = self.acfg.search.invalid_metric_upper_bound if self.acfg.search.invalid_metric_upper_bound else upper_bound
        v1 = self.best_metric
        v2 = node.metric.value
        if v1 is None or v2 is None:
            return True
        elif v1 == 0 or v2 == 0:
            return abs(v1 - v2) <= upper_bound
        else:
            ratio = max(abs(v1), abs(v2)) / min(abs(v1), abs(v2))
            return ratio <= upper_bound

    def _copy_to_submission_dir(self, submission_file_path):
        """Copy submission directly to SUBMISSION_DIR for extraction resilience."""
        from pathlib import Path as _Path
        try:
            submission_dir = _Path(os.environ.get("SUBMISSION_DIR", "/home/submission"))
            submission_dir.mkdir(exist_ok=True, parents=True)
            shutil.copy(submission_file_path, submission_dir / "submission.csv")
            logger.info(f"Copied best submission to {submission_dir}/submission.csv")
        except Exception as e:
            logger.warning(f"Failed to copy submission to SUBMISSION_DIR: {e}")

    def _is_timeout_node(self, node: MCTSNode) -> bool:
        """Check if node failed due to timeout (unfixable by debugging)."""
        if node.exc_type and "timeout" in node.exc_type.lower():
            return True
        output = node.term_out or ""
        timeout_patterns = [
            "timeouterror: execution exceeded",
            "exceeded the time limit",
            "killed due to timeout",
            "timed out",
            "process killed",
            "oom",
            "out of memory",
            "repl child process failed",
        ]
        output_lower = output.lower()
        return any(p in output_lower for p in timeout_patterns)

    def _is_dead_bug(self, node: MCTSNode) -> bool:
        """Check for 0% debug success rate, no traceback, short exec, short output."""
        if not node.is_buggy:
            return False
        term = node.term_out or ""
        if (
            node.exc_type is None
            and (node.exec_time is not None and node.exec_time < 5)
            and len(term) < 100
        ):
            return True
        return False

    def _step_search(self, parent_node: MCTSNode, exec_callback: ExecCallbackType):
        logger.info(f"[_step_search] Processing node: {parent_node.id}")
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")
        result_node = None
        _root = False
    
        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    result_node = self._draft()
                    result_node.lock = True
                    logger.info(f"[_step_search]Draft node {result_node.id} is locked.")
                elif parent_node.is_buggy or parent_node.is_valid is False:
                    # Mark timeout/dead bug nodes as terminal so MCTS stops expanding them
                    if self._is_timeout_node(parent_node) or self._is_dead_bug(parent_node):
                        logger.info(f"Marking timeout/dead node {parent_node.id} as terminal.")
                        parent_node.is_terminal = True
                    else:
                        result_node = self._debug(parent_node)
                elif parent_node.is_buggy is False:
                    result_node = self._improve(parent_node)
                else:
                    logger.warning(f"[_step_search] node {parent_node.id} is_buggy is None.")
                
                if result_node:
                
                    exe_res = exec_callback(result_node.code, result_node.id, True)
                
                    result_node = self.parse_exec_result(
                        node=result_node,
                        exec_result=exe_res
                    )

                    # Post-debug subsampling cleanup: if debug succeeded but code has
                    # debug subsampling, strip it and re-run on full data
                    if (not result_node.is_buggy) and result_node.stage == "debug":
                        import re as _re
                        _subsample_patterns = [
                            r"\.sample\s*\(", r"\bfrac\s*=", r"\bnrows\s*=",
                            r"\bhead\s*\(", r"DEBUG:.*subsample", r"DEBUG:.*sample", r"#\s*DEBUG",
                        ]
                        if any(_re.search(p, result_node.code, flags=_re.IGNORECASE) for p in _subsample_patterns):
                            logger.info("Debug node succeeded with subsampling; cleaning and re-running on full data.")
                            _clean_prompt = {
                                "Instruction": (
                                    "You are cleaning debug code. Remove any debug subsampling, sampling fractions, "
                                    "nrows/head truncations, or DEBUG placeholders. Preserve all other logic, comments, "
                                    "and structure. Return ONLY the cleaned Python code in a single markdown code block."
                                ),
                                "Code": wrap_code(result_node.code),
                            }
                            _cleaned_text = query(
                                system_message=_clean_prompt, user_message=None,
                                model=self.acfg.code.model, temperature=0,
                                cfg=self.cfg,
                            )
                            _cleaned_code = extract_code(_cleaned_text)
                            if _cleaned_code:
                                result_node.code = _cleaned_code
                                _full_exec = exec_callback(_cleaned_code, result_node.id, True)
                                result_node = self.parse_exec_result(node=result_node, exec_result=_full_exec)

                    # Rename any stray submission csv to the expected hash filename
                    expected_sub = self.cfg.workspace_dir / "submission" / f"submission_{result_node.id}.csv"
                    if not expected_sub.exists():
                        sub_dir = self.cfg.workspace_dir / "submission"
                        csvs = glob.glob(str(sub_dir / "*.csv"))
                        if csvs:
                            newest = max(csvs, key=os.path.getmtime)
                            os.rename(newest, str(expected_sub))
                            logger.info(f"Renamed {os.path.basename(newest)} -> submission_{result_node.id}.csv")

                    # Header coercion: fix column names/order to match sample_submission.csv
                    if expected_sub.exists():
                        sample_sub = self.cfg.workspace_dir / "input" / "sample_submission.csv"
                        if sample_sub.exists():
                            try:
                                exp_cols = _read_csv_header(sample_sub)
                                tmp_coerced, coerce_err = _maybe_coerce_submission_header(expected_sub, exp_cols)
                                if tmp_coerced is not None:
                                    shutil.move(str(tmp_coerced), str(expected_sub))
                                    logger.info(f"Coerced submission header for node {result_node.id} to match sample_submission.csv")
                                elif coerce_err:
                                    logger.debug(f"Header coercion skipped for node {result_node.id}: {coerce_err}")
                            except Exception as e:
                                logger.debug(f"Header coercion failed for node {result_node.id}: {e}")

                    if not result_node.is_buggy:
                        if not expected_sub.exists():
                            result_node.is_buggy = True
                            result_node.metric = WorstMetricValue()
                            logger.info(f"Actually, node {result_node.id} did not produce a submission.csv")
                    logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                    if not self.check_metric_valid(node=result_node):
                        result_node.metric = WorstMetricValue()
                        logger.info(f"node {result_node.id} generate invalid metric.")
                    result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
                    if parent_node.is_buggy and result_node.is_buggy is False:
                        parent_node.is_debug_success = True
                    
                    _root = self.check_improvement(result_node, parent_node)
                    with self.journal_lock:
                        self.journal.append(result_node)
                        # Learn from bugs AFTER step is assigned by journal.append
                        if self.bug_consultant:
                            try:
                                self.bug_consultant.learn_from_bug(result_node, journal=self.journal)
                                logger.info(f"Bug consultant ingested node {result_node.id}, step={result_node.step}")
                            except Exception as e:
                                logger.warning(f"Bug consultant learn_from_bug failed: {e}")


            except Exception as e:
                logger.warning("Current node generation failed, rolling back to unlock the draft node.")
                self.backpropagate(node=parent_node, value=0, add_to_tree=False)
                parent_node.sub_expected_child_count()
                raise e

        else:
            logger.info(f"current node is terminal, backpropagating!!")
            self.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node
    
    def get_best_node(self, node_list):
        good_node = [n for n in node_list if not n.is_buggy and n.metric]
        if not good_node:
            return None
        return max(good_node, key=lambda n: n.metric)

    def step(self, node: MCTSNode, exec_callback: ExecCallbackType) -> bool:   
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            self.search_start_time = time.time()

        if not node or node.stage == "root":
            node = self.select(self.virtual_root)

        _root, result_node = self._step_search(node, exec_callback=exec_callback)
        if result_node:
            submission_file_path = self.cfg.workspace_dir / "submission" / f"submission_{result_node.id}.csv"
            logger.info(f"In the search step from node {node.id}, the generated node is {result_node.id}, the metric is {result_node.metric.value}")
        if result_node and result_node.metric.value is not None:
            if self.best_node is None or self.best_node.metric < result_node.metric:
                logger.info(f"Node {result_node.id} is the best node so far")
                if self.best_node is None or result_node.is_valid is not False:
                    self.best_node = result_node
                    self.best_metric = result_node.metric.value
                    logger.info(f"Updated best_metric to {self.best_metric}")
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    with self.save_node_lock:
                        best_solution_dir.mkdir(exist_ok=True, parents=True)
                        best_submission_dir.mkdir(exist_ok=True, parents=True)
                        shutil.copy(
                            submission_file_path,
                            best_submission_dir / "submission.csv",
                        )
                        # Also write directly to SUBMISSION_DIR for extraction resilience
                        self._copy_to_submission_dir(submission_file_path)
                        with open(best_solution_dir / "solution.py", "w") as f:
                            f.write(result_node.code)
                        with open(best_solution_dir / "node_id.txt", "w") as f:
                            f.write(str(result_node.id))
                else:
                    logger.info(f"Node {result_node.id} is a invalid node")
                    logger.info(f"Node {self.best_node.id} is still the best node")
            else:
                if self.best_node.is_valid is False:
                    logger.info(f"Node {self.best_node.id} is invalid, {result_node.id} is the best node so far")
                    self.best_node = result_node
                    self.best_metric = result_node.metric.value
                    logger.info(f"Updated best_metric to {self.best_metric}")
                    best_solution_dir = self.cfg.workspace_dir / "best_solution"
                    best_submission_dir = self.cfg.workspace_dir / "best_submission"
                    with self.save_node_lock:
                        best_solution_dir.mkdir(exist_ok=True, parents=True)
                        best_submission_dir.mkdir(exist_ok=True, parents=True)
                        shutil.copy(
                            submission_file_path,
                            best_submission_dir / "submission.csv",
                        )
                        # Also write directly to SUBMISSION_DIR for extraction resilience
                        self._copy_to_submission_dir(submission_file_path)
                        with open(best_solution_dir / "solution.py", "w") as f:
                            f.write(result_node.code)
                        with open(best_solution_dir / "node_id.txt", "w") as f:
                            f.write(str(result_node.id))

                else:
                    logger.info(f"Node {result_node.id} is not the best node")
                    logger.info(f"Node {self.best_node.id} is still the best node")
        elif not result_node:
            logger.info(f"Result node is None.")
        else:
            logger.info(f"result node has bug.")
        if self.best_node:
            logger.info(f"Best metric value is {self.best_node.metric.value}.")

        if not self.acfg.save_all_submission and result_node and os.path.exists(submission_file_path):
            os.remove(submission_file_path)
        self.current_step = len(self.journal)
        if _root or result_node is None:
            logger.info(f"agent return root to main")
            return self.virtual_root
        else:
            logger.info(f"agent return {result_node.id} to main")
            return result_node
        