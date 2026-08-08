import hashlib
import logging
import math
import random
import re
from pathlib import Path
from typing import Any, Callable, cast

import humanize
import numpy as np
from .backend import query
from .bug_consultant import BugConsultant
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .reviewer import Reviewer
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.metrics_io import normalize_metrics, parse_aide_metrics
from .utils.mlebench_grading import _maybe_coerce_submission_header, _read_csv_header
from .utils.response import extract_code, extract_text_up_to_code, wrap_code
from .utils.stage_manager import StageInfo, StageManager

logger = logging.getLogger("aide")


ExecCallbackType = Callable[[str, bool], ExecutionResult]


class Agent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None
        self.stage_manager: StageManager | None = None
        self.current_stage: StageInfo | None = None

        self._last_actor_context: dict[str, Any] | None = None

        # Progressive complexity (prompt-only).
        if getattr(self.cfg, "progressive", None) is not None and self.cfg.progressive.enabled:
            self.stage_manager = StageManager(self.cfg.progressive, total_steps=self.acfg.steps)

        # Bug consultant (memory writer + retrieval-only reader).
        self.bug_consultant: BugConsultant | None = None
        if getattr(self.acfg.search, "use_bug_consultant", False):
            save_dir = (self.cfg.log_dir / "bug_consultant").resolve()
            self.bug_consultant = BugConsultant(
                model=self.acfg.feedback.model,
                temperature=0.3,
                save_dir=save_dir,
                max_bug_records=int(getattr(self.acfg.search, "max_bug_records", 500) or 500),
                advice_budget_chars=int(getattr(self.acfg.search, "advice_budget_chars", 200000) or 200000),
                max_active_bugs=int(getattr(self.acfg.search, "max_active_bugs", 200) or 200),
                max_trials_per_bug=int(getattr(self.acfg.search, "max_trials_per_bug", 20) or 20),
                delete_pruned_bug_files=bool(getattr(self.acfg.search, "delete_pruned_bug_files", False)),
            )
            # Resume support: rebuild memory from existing journal.
            if self.journal.nodes:
                self.bug_consultant.ingest_journal(self.journal)

        # Context-aware reviewer (schema-compatible with legacy reviewer).
        self.reviewer = Reviewer(model=self.acfg.feedback.model, temperature=self.acfg.feedback.temp)

    def _format_bug_node_for_memory(self, node: Node) -> str:
        lines = [f"**Node #{node.step}**"]
        if node.plan:
            lines.append(f"Plan: {node.plan}")
        if node.exc_type:
            lines.append(f"Error: {node.exc_type}")
        if node.analysis:
            lines.append(f"Reviewer suggestion: {node.analysis}")
        if node.code:
            lines.append(f"Code:\n```python\n{node.code}\n```")
        out = node.term_out
        if out:
            lines.append(f"Output:\n```\n{out}\n```")
        return "\n\n".join(lines)

    def _is_dead_bug(self, node: Node) -> bool:
        """
        Detect "dead bugs" that have 0% debug success rate.
        These are quick crashes with no traceback - nothing to debug.
        Skip debugging and draft fresh instead.
        """
        # If there's an exception type, we have something to debug
        if node.exc_type is not None:
            return False
        # If execution took more than 5 seconds, something ran
        if node.exec_time is not None and node.exec_time >= 5:
            return False
        # If there's substantial output, we have something to work with
        term_out = node.term_out or ""
        if len(term_out) > 100:
            return False
        # Dead bug: no exception, quick crash, minimal output
        return True

    def _buggy_nodes_for_memory(self, *, exclude_step: int | None = None) -> list[str]:
        """
        Return formatted buggy nodes for prompt memory.
        Controlled by:
        - agent.search.bug_context_count
        """
        count = int(getattr(self.acfg.search, "bug_context_count", 0) or 0)
        if count == 0:
            return []

        buggy_nodes = [n for n in self.journal.buggy_nodes if n.step is not None]
        buggy_nodes = sorted(buggy_nodes, key=lambda n: n.step)
        if exclude_step is not None:
            buggy_nodes = [n for n in buggy_nodes if n.step != exclude_step]

        if count > 0:
            buggy_nodes = buggy_nodes[-count:]

        return [self._format_bug_node_for_memory(n) for n in buggy_nodes]

    def _get_stage_prompt_section(self) -> str:
        if not self.stage_manager or not self.current_stage:
            return ""

        stage = self.current_stage
        # Respect global CV preference, but cap by stage.
        cv_cfg = int(getattr(self.acfg, "k_fold_validation", 1))
        if cv_cfg <= 1:
            cv_use = 1
        else:
            cv_use = min(cv_cfg, int(stage.max_cv_folds))

        cv_line = (
            "Do not use cross-validation unless clearly appropriate."
            if cv_use <= 1
            else f"Use {cv_use}-fold cross-validation."
        )

        pct = int(round(stage.data_fraction * 100))
        return (
            f"**COMPUTATIONAL STAGE: {stage.name.upper()}** (Step {stage.step}/{stage.total_steps})\n\n"
            f"**Data Sampling**: {stage.data_instruction}\n"
            f"- Target fraction: {pct}%\n\n"
            f"**Hyperparameter Strategy**: {stage.hparam_instruction}\n\n"
            f"**Cross-Validation**: {cv_line} {stage.cv_folds_instruction}\n\n"
            "**CRITICAL**: Follow these stage instructions exactly."
        )

    def _is_timeout_node(self, node: Node) -> bool:
        """Check if node failed due to timeout (unfixable by debugging)."""
        # Check exception type
        if node.exc_type and "timeout" in node.exc_type.lower():
            return True
        # Check output for timeout indicators
        # NOTE: "execution time: a moment seconds" means QUICK crash (< 1 sec), NOT timeout!
        # NOTE: "time limit is" appears in ALL execution messages, so don't match on that
        output = node.term_out or ""
        timeout_patterns = [
            "timeouterror: execution exceeded",  # Actual timeout from interpreter
            "exceeded the time limit",           # Actual timeout message
            "killed due to timeout",
            "timed out",
            "process killed",
            "oom",                               # Out of memory
            "out of memory",
        ]
        output_lower = output.lower()
        return any(p in output_lower for p in timeout_patterns)

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth + NOT timeout
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth
                    and not self._is_timeout_node(n))
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return greedy_node

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
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **print the value of the evaluation metric computed on a hold-out validation set**.",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
            "DO NOT use GridSearchCV or RandomizedSearchCV.",
            "**Early Stopping**: For LightGBM, use `callbacks=[lgb.early_stopping(stopping_rounds=N)]`. For XGBoost, use native API: `xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)`.",
            "Compute safety: avoid unconstrained parallelism. Do NOT use `n_jobs=-1` anywhere. If you set `n_jobs` / `thread_count` / `num_threads`, cap them to `int(os.getenv('AIDE_NUM_THREADS', '{n}'))` (use the same cap consistently).".format(
                n=int(getattr(self.cfg.exec, "num_threads", 4) or 4)
            ),
            'All the provided input data is stored in "./input" directory.',
            '**If there is test data provided for this task, please save the test predictions in a `submission.csv` file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "**CRITICAL - Data Leakage Prevention**: NEVER use target encoding or create features using the target variable (e.g., target mean encoding, target-based statistics, label encoding with target information). This causes severe data leakage and inflates validation scores artificially. Any feature engineering must use only the input features, never the target.",
            "**CRITICAL - Hyperparameter Optimization**: NEVER optimize hyperparameters "
            "(ensemble weights, temperature scaling, threshold tuning, blending coefficients) "
            "by evaluating on training set labels directly (e.g., log_loss(y_train, preds)). "
            "This causes severe data leakage. All tuning must use cross-validation folds or "
            "a held-out validation set.",
            "At the end of the script, print a single-line JSON with prefix `AIDE_METRICS_JSON=` so the harness can parse metrics. Include: `valid`, `lower_is_better`, and (for CV) `cv_mean`, `cv_std`, `cv_folds` (list of fold scores).",
            "If a `Stage Instructions` section is provided, follow it exactly for data fraction, hyperparameter tuning, and CV folds (it overrides generic defaults).",
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

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        constraint_bits: list[str] = []
        plan_cfg = getattr(self.cfg, "plan_constraints", None)
        if plan_cfg is not None and getattr(plan_cfg, "enabled", False):
            max_sentences = getattr(plan_cfg, "max_sentences", None)
            if max_sentences is not None:
                constraint_bits.append(f"≤{int(max_sentences)} sentences")
        constraint_part = f" ({'; '.join(constraint_bits)})" if constraint_bits else ""
        return {
            "Response format": (
                f"Your response should be an outline/sketch of your proposed solution in natural language{constraint_part}, "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def _solution_sketch_length_guideline(self) -> str | None:
        plan_cfg = getattr(self.cfg, "plan_constraints", None)
        if plan_cfg is None or not getattr(plan_cfg, "enabled", False):
            return None

        max_sentences = getattr(plan_cfg, "max_sentences", None)
        parts: list[str] = []
        if max_sentences is not None:
            parts.append(f"at most {int(max_sentences)} sentences")
        if not parts:
            return None
        return f"The solution sketch should be {', '.join(parts)}."

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        stage_section = self._get_stage_prompt_section()
        exec_summary = self.bug_consultant.get_prevention_guidance(mode="executive", journal=self.journal) if self.bug_consultant else ""
        bug_context_mode = str(getattr(self.acfg.search, "bug_context_mode", "consultant"))
        inject_shared_context = bool(getattr(self.acfg.search, "inject_shared_context", True))

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "In order to win this competition, you need to come up with an excellent and creative plan "
                "for a solution and then implement this solution in Python. We will now provide a description of the task."
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        if stage_section:
            prompt["Stage Instructions"] = stage_section
        if exec_summary and bug_context_mode in ("consultant", "both") and inject_shared_context:
            prompt["Bug Prevention Alert"] = exec_summary

        prompt["Instructions"] |= self._prompt_resp_fmt

        sketch_guideline = [
            "Keep the initial solution design relatively simple and robust; follow the Stage Instructions for subsampling, and CV folds.",
            "Take the Memory section into consideration when proposing the design,"
            " don't propose the same modelling solution but keep the evaluation the same.",
            "Do not subsample the dataset in draft runs; always use the full training data.",
            "NEVER use target encoding or any feature engineering that involves the target variable - this causes data leakage.",
            "Propose an evaluation metric that is reasonable for this task.",
            "Don't suggest to do EDA.",
            "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
        ]
        length_guideline = self._solution_sketch_length_guideline()
        if length_guideline:
            sketch_guideline.insert(3, length_guideline)
        prompt["Instructions"] |= {"Solution sketch guideline": sketch_guideline}
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        self._last_actor_context = {
            "stage": stage_section,
            "bug_prevention": exec_summary if bug_context_mode in ("consultant", "both") and inject_shared_context else "",
        }
        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        stage_section = self._get_stage_prompt_section()
        exec_summary = self.bug_consultant.get_prevention_guidance(mode="executive", journal=self.journal) if self.bug_consultant else ""
        bug_context_mode = str(getattr(self.acfg.search, "bug_context_mode", "consultant"))
        inject_shared_context = bool(getattr(self.acfg.search, "inject_shared_context", True))

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
                "solution below and should improve it in order to further increase the (test time) performance. "
                "For this you should first outline a plan in natural language for how the solution can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }
        if stage_section:
            prompt["Stage Instructions"] = stage_section
        if exec_summary and bug_context_mode in ("consultant", "both") and inject_shared_context:
            prompt["Bug Prevention Alert"] = exec_summary

        prompt["Instructions"] |= self._prompt_resp_fmt

        improve_guideline = [
            "The solution sketch should be a natural language description of how the previous solution can be improved.",
            "You should be very specific and propose concrete improvements that you will actually implement.",
            "Take the Memory section into consideration when proposing the improvement.",
            "If there is any debug subsampling code in the previous solution (e.g., data sampling, `frac=...`, `nrows=...`, `head(...)`, `# DEBUG`), remove it and use the full dataset.",
            "NEVER add target encoding or any feature engineering that involves the target variable - this causes data leakage.",
            "Don't suggest to do EDA.",
        ]
        length_guideline = self._solution_sketch_length_guideline()
        if length_guideline:
            improve_guideline.insert(4, length_guideline)
        prompt["Instructions"] |= {"Solution improvement sketch guideline": improve_guideline}
        prompt["Instructions"] |= self._prompt_impl_guideline

        self._last_actor_context = {
            "stage": stage_section,
            "bug_prevention": exec_summary if bug_context_mode in ("consultant", "both") and inject_shared_context else "",
        }
        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _debug(self, parent_node: Node) -> Node:
        stage_section = self._get_stage_prompt_section()
        # Debug gets RAG context (similar bugs) not general prevention guidance
        bug_context_mode = str(getattr(self.acfg.search, "bug_context_mode", "consultant"))
        inject_shared_context = bool(getattr(self.acfg.search, "inject_shared_context", True))
        debug_history = ""
        active_trials = ""
        if self.bug_consultant:
            try:
                # Use full output (not truncated) for better bug matching
                full_error_msg = "".join(parent_node._term_out) if hasattr(parent_node, "_term_out") else parent_node.term_out
                retrieval = self.bug_consultant.retrieve_relevant_context(
                    current_error_type=parent_node.exc_type or "Unknown",
                    current_error_msg=full_error_msg,
                    current_code=parent_node.code,
                    original_plan=parent_node.plan or "",
                )
                debug_history = self.bug_consultant.format_context_for_actor(retrieval)
            except Exception:
                debug_history = ""
            try:
                active_trials = self.bug_consultant.format_active_trial_history(
                    f"bug_{parent_node.step}", max_trials=5
                )
            except Exception:
                active_trials = ""

        past_buggy_nodes = []
        if bug_context_mode in ("buggy_code", "both"):
            past_buggy_nodes = self._buggy_nodes_for_memory(exclude_step=parent_node.step)

        prompt: Any = {
            "Introduction": (
                "You are a Kaggle grandmaster attending a competition. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
        }
        # Move Bug Context to TOP of prompt for better visibility
        if debug_history and bug_context_mode in ("consultant", "both") and inject_shared_context:
            prompt["Historical Bug Context (Curated)"] = debug_history
        prompt["Task description"] = self.task_desc
        prompt["Approved Solution Plan"] = parent_node.plan or ""
        prompt["Previous (buggy) implementation"] = wrap_code(parent_node.code)
        prompt["Execution output"] = wrap_code(parent_node.term_out, lang="")
        prompt["Instructions"] = {}
        if stage_section:
            prompt["Stage Instructions"] = stage_section
        if active_trials and inject_shared_context:
            prompt["Current Bug Trial History"] = active_trials
        if past_buggy_nodes:
            prompt["Past buggy nodes (raw)"] = past_buggy_nodes

        prompt["Instructions"] |= self._prompt_resp_fmt
        blocklist_check = (
            "BLOCKLIST CHECK: Read 'Historical Bug Context' and 'Current Bug Trial History' - those approaches have ALREADY FAILED and WILL CRASH AGAIN if you use them. You MUST use a DIFFERENT approach."
            if inject_shared_context else
            "You must try a DIFFERENT approach from any previously attempted fixes shown in this prompt."
        )
        bugfix_guideline = [
            blocklist_check,
            "You should write a natural language description of how the issue in the previous implementation can be fixed.",
            "**FOR QUICK DEBUGGING** (ONLY for datasets >50,000 rows): You may subsample to 10%, but MUST check class counts first to avoid crashes:\n```python\nmin_class_count = y.value_counts().min()\nn_splits = 5\nif min_class_count >= n_splits:\n    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)\nelse:\n    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # FALLBACK\n```\nNEVER use StratifiedKFold on subsampled data without checking class counts! Add `# DEBUG: Using 10% subsample` comment so it can be removed once fixed.",
            "Don't suggest to do EDA.",
        ]
        length_guideline = self._solution_sketch_length_guideline()
        if length_guideline:
            bugfix_guideline.insert(1, length_guideline)
        prompt["Instructions"] |= {"Bugfix improvement sketch guideline": bugfix_guideline}
        prompt["Instructions"] |= {
            "Plan adherence": [
                "Honor the IDEA of the Approved Solution Plan (model choice, CV strategy, feature approach)",
                "Implementation DETAILS (API calls, parameter names) CAN change to avoid bugs",
                "Example: Plan says 'early stopping' → choose a working implementation (e.g., remove the feature or use alternative method)",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        self._last_actor_context = {
            "stage": stage_section,
            "historical_bug_context": debug_history if bug_context_mode in ("consultant", "both") and inject_shared_context else "",
            "past_buggy_nodes": past_buggy_nodes,
            "approved_solution_plan": parent_node.plan or "",
        }

        # Log and save debug context being passed
        logger.info("=== DEBUG NODE CONTEXT ===")
        debug_context_parts = []
        if debug_history and bug_context_mode in ("consultant", "both") and inject_shared_context:
            logger.info("[Historical Bug Context]:\n%s", debug_history)
            debug_context_parts.append(f"## Historical Bug Context\n{debug_history}")
        if active_trials and inject_shared_context:
            logger.info("[Current Bug Trial History]:\n%s", active_trials)
            debug_context_parts.append(f"## Current Bug Trial History\n{active_trials}")
        logger.info("=== END DEBUG NODE CONTEXT ===")

        # Save debug context to file
        if self.bug_consultant and self.bug_consultant.save_dir and debug_context_parts:
            try:
                debug_file = self.bug_consultant.save_dir / f"debug_context_step_{parent_node.step}.md"
                debug_file.write_text("\n\n".join(debug_context_parts))
            except Exception:
                pass

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        step_idx = len(self.journal.nodes)
        if self.stage_manager:
            prev = self.current_stage.name if self.current_stage else None
            self.current_stage = self.stage_manager.get_stage(step_idx)
            if prev != self.current_stage.name:
                logger.info(
                    "🎯 Progressive stage: %s (step %s/%s)",
                    self.current_stage.name,
                    step_idx,
                    self.acfg.steps,
                )

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            # Skip debugging for "dead bugs" - no traceback, quick crash, nothing to debug
            if self._is_dead_bug(parent_node):
                logger.info("Skipping debug for dead bug (no traceback, quick crash) - drafting instead")
                result_node = self._draft()
            else:
                result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        # Assign a deterministic step before execution so downstream writers can use it.
        result_node.step = step_idx
        if self.current_stage is not None:
            result_node.progressive_stage = self.current_stage.name
            result_node.data_fraction_used = self.current_stage.data_fraction
            result_node.hparam_tuning_enabled = not self.current_stage.disable_hparam_tuning

        actor_context = self._last_actor_context
        exec_res = exec_callback(result_node.code, True)
        self.parse_exec_result(node=result_node, exec_result=exec_res, actor_context=actor_context)

        # Old-way debug subsampling: allow fast subsampled debug runs, but if the debug node succeeds,
        # remove debug subsampling and re-run on full data to validate and log the true metric.
        if (not result_node.is_buggy) and result_node.stage_name == "debug":
            patterns = [
                r"\.sample\s*\(",
                r"\bfrac\s*=",
                r"\bnrows\s*=",
                r"\bhead\s*\(",
                r"DEBUG:.*subsample",
                r"DEBUG:.*sample",
                r"#\s*DEBUG",
            ]

            def has_debug_subsampling(code: str) -> bool:
                return any(re.search(pat, code, flags=re.IGNORECASE) for pat in patterns)

            def strip_debug_subsampling(code: str) -> str:
                prompt = {
                    "Instruction": (
                        "You are cleaning debug code. Remove any debug subsampling, sampling fractions, "
                        "nrows/head truncations, or DEBUG placeholders. Preserve all other logic, comments, "
                        "and structure. Return ONLY the cleaned Python code in a single markdown code block."
                    ),
                    "Code": wrap_code(code),
                }
                cleaned = query(
                    system_message=prompt,
                    user_message=None,
                    model=self.acfg.code.model,
                    temperature=0,
                )
                cleaned_code = extract_code(cleaned)
                return cleaned_code if cleaned_code else code

            if has_debug_subsampling(result_node.code):
                logger.info("Debug node succeeded; cleaning debug subsampling and re-running on full data.")
                cleaned = strip_debug_subsampling(result_node.code)
                result_node.code = cleaned
                full_exec = exec_callback(cleaned, True)
                self.parse_exec_result(node=result_node, exec_result=full_exec, actor_context=actor_context)

        self.journal.append(result_node)

        # Timing metadata (optional).
        if getattr(self.cfg, "timing", None) is not None and self.cfg.timing.track_cumulative_time:
            prev_cum = 0.0
            if len(self.journal.nodes) > 1:
                prev_cum = float(getattr(self.journal.nodes[-2], "cumulative_exec_time", 0.0) or 0.0)
            exec_time = float(getattr(result_node, "exec_time", 0.0) or 0.0)
            result_node.cumulative_exec_time = prev_cum + exec_time

        # Update debugging consultant memory (writer).
        if self.bug_consultant:
            self.bug_consultant.learn_from_bug(result_node, journal=self.journal)

    def parse_exec_result(
        self, node: Node, exec_result: ExecutionResult, actor_context: dict[str, Any] | None = None
    ):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        # Pass FULL output to reviewer (not truncated) so LLM can extract cv_folds reliably
        full_term_out = "".join(node._term_out)
        response = cast(
            dict,
            self.reviewer.review(
                task_desc=self.task_desc,
                code=node.code,
                term_out=full_term_out,
                actor_context=actor_context,
            ),
        )

        # Prefer structured metrics printed by the runfile when available.
        # Use FULL output for JSON parsing (not truncated) to avoid cutting JSON line mid-string
        parsed = parse_aide_metrics(full_term_out)
        norm = normalize_metrics(parsed) if parsed is not None else None
        if norm is not None:
            valid = norm.get("valid")
            if isinstance(valid, (int, float)) and not isinstance(valid, bool):
                response["metric"] = float(valid)
            lower_is_better = norm.get("lower_is_better")
            if isinstance(lower_is_better, bool):
                response["lower_is_better"] = lower_is_better

        # Coerce metric to float, otherwise treat as missing.
        metric = response.get("metric")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            response["metric"] = None
        else:
            metric_f = float(metric)
            response["metric"] = metric_f if math.isfinite(metric_f) else None

        # Guardrail: treat exact 0/1 metrics as invalid placeholders and mark buggy.
        if response["metric"] in (0.0, 1.0):
            response["metric"] = None
            response["is_bug"] = True

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )

        # ---- standardized metric logging (does not affect search behavior) ----
        # Always store the reviewer-reported validation metric as a default.
        node.valid_metric = response["metric"] if isinstance(response["metric"], float) else None

        # Two-layer approach: Try JSON parsing first, then LLM extraction as fallback
        # cv_folds is SOURCE OF TRUTH - always calculate cv_mean and cv_std from folds

        def validate_cv_folds(folds: list[float]) -> tuple[bool, str | None]:
            """
            Check if CV folds are valid (not placeholder values like all 0s or all 1s).
            Returns: (is_valid, error_message)
            """
            if not folds or len(folds) == 0:
                return False, None
            # Check if all folds are exactly 0.0 or 1.0 (placeholder values)
            if all(f == 0.0 for f in folds):
                msg = f"Invalid CV folds: all scores are 0.0 {folds}. This indicates the model failed to make meaningful predictions or the evaluation metric is broken. Please check the model training, feature preprocessing, and metric calculation."
                logger.warning(msg)
                return False, msg
            if all(f == 1.0 for f in folds):
                msg = f"Invalid CV folds: all scores are 1.0 {folds}. This indicates either perfect placeholder values or a broken evaluation metric. Please verify the metric calculation and ensure it reflects actual model performance."
                logger.warning(msg)
                return False, msg
            # Check if all folds are identical (suspicious, likely placeholder)
            if len(set(folds)) == 1:
                msg = f"Suspicious CV folds: all {len(folds)} folds have identical score {folds[0]}. Real cross-validation should show variance across folds. This suggests a bug in the CV loop, metric calculation, or the use of placeholder values. Please verify the CV implementation."
                logger.warning(msg)
                return False, msg
            return True, None

        # Track validation errors to update node analysis if cv_folds are invalid
        cv_validation_error = None

        # Layer 1: Try JSON parsing (primary, fast and structured)
        if norm is not None and isinstance(norm.get("cv_folds"), list):
            try:
                cv_folds_candidate = [float(v) for v in norm["cv_folds"]]
                is_valid, error_msg = validate_cv_folds(cv_folds_candidate)
                if is_valid:
                    node.cv_folds = cv_folds_candidate
                    # Calculate mean and std FROM folds (source of truth)
                    node.cv_mean = float(np.mean(node.cv_folds))
                    node.cv_std = float(np.std(node.cv_folds))
                    logger.info(f"CV metrics from JSON: mean={node.cv_mean:.6f}, std={node.cv_std:.6f}, folds={len(node.cv_folds)}")
                else:
                    node.cv_folds = None
                    cv_validation_error = error_msg
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse cv_folds from JSON: {e}")
                node.cv_folds = None

        # Layer 2: Fallback to LLM-extracted cv_folds (handles any format)
        if node.cv_folds is None and cv_validation_error is None:
            cv_folds_raw = response.get("cv_folds")
            if isinstance(cv_folds_raw, list) and len(cv_folds_raw) > 0:
                try:
                    cv_folds_candidate = [float(v) for v in cv_folds_raw]
                    is_valid, error_msg = validate_cv_folds(cv_folds_candidate)
                    if is_valid:
                        node.cv_folds = cv_folds_candidate
                        # Calculate mean and std FROM folds (source of truth)
                        node.cv_mean = float(np.mean(node.cv_folds))
                        node.cv_std = float(np.std(node.cv_folds))
                        logger.info(f"CV metrics from LLM: mean={node.cv_mean:.6f}, std={node.cv_std:.6f}, folds={len(node.cv_folds)}")
                    else:
                        node.cv_folds = None
                        cv_validation_error = error_msg
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse cv_folds from LLM response: {e}")
                    node.cv_folds = None

        # If cv_folds validation failed, mark as buggy and update analysis
        if cv_validation_error is not None:
            node.is_buggy = True
            node.metric = WorstMetricValue()
            # Prepend validation error to existing analysis
            if node.analysis:
                node.analysis = f"[CV VALIDATION ERROR] {cv_validation_error}\n\nOriginal analysis: {node.analysis}"
            else:
                node.analysis = f"[CV VALIDATION ERROR] {cv_validation_error}"
            logger.error(f"Node {node.step} marked as buggy due to invalid cv_folds")

        # STRICT ENFORCEMENT: cv_folds is the SOURCE OF TRUTH
        # Mark node as buggy if no valid cv_folds found (no fallback to single validation)
        if node.cv_folds is None:
            node.is_buggy = True
            node.metric = WorstMetricValue()
            error_msg = (
                "Missing CV folds: The code did not report cross-validation fold scores. "
                f"Expected {self.acfg.k_fold_validation}-fold CV with all fold scores in cv_folds list. "
                "Ensure the code prints AIDE_METRICS_JSON with cv_folds=[...] containing all fold scores."
            )
            if node.analysis:
                node.analysis = f"[MISSING CV FOLDS] {error_msg}\n\nOriginal analysis: {node.analysis}"
            else:
                node.analysis = f"[MISSING CV FOLDS] {error_msg}"
            # Set cv_mean/cv_std to None to make it clear there's no valid CV data
            node.cv_mean = None
            node.cv_std = None
            logger.error(f"Node {node.step} marked as buggy due to missing cv_folds")

        # Save submission.csv immediately after execution (before it gets overwritten by next node)
        if hasattr(self.cfg, "export") and self.cfg.export.save_submissions:
            submission_src = Path(self.cfg.workspace_dir) / "working" / "submission.csv"
            if submission_src.exists():
                solutions_dir = Path(self.cfg.log_dir) / "solutions"
                solutions_dir.mkdir(parents=True, exist_ok=True)
                submission_dst = solutions_dir / f"submission_node_{node.step}.csv"
                if not submission_dst.exists():
                    adjusted_src = submission_src
                    tmp_adjusted: Path | None = None
                    sample_submission = Path(self.cfg.workspace_dir) / "input" / "sample_submission.csv"
                    expected_cols: list[str] | None = None
                    if sample_submission.exists():
                        try:
                            expected_cols = _read_csv_header(sample_submission)
                        except Exception as e:  # pragma: no cover - best-effort logging only
                            logger.debug(
                                "Failed to read sample submission header (%s): %s",
                                sample_submission,
                                e,
                            )
                    if expected_cols:
                        tmp_adjusted, coerce_err = _maybe_coerce_submission_header(
                            submission_src, expected_cols
                        )
                        if coerce_err is not None:
                            logger.debug(
                                "Submission schema auto-coercion skipped: %s",
                                coerce_err,
                            )
                        elif tmp_adjusted is not None:
                            adjusted_src = tmp_adjusted
                            logger.debug(
                                "Submission schema adjusted to match sample header: %s",
                                sample_submission,
                            )
                    submission_dst.write_bytes(adjusted_src.read_bytes())
                    if tmp_adjusted is not None and tmp_adjusted.exists():
                        tmp_adjusted.unlink(missing_ok=True)
                    try:
                        node.submission_csv_path = str(submission_dst)
                        # Calculate SHA256 hash
                        h = hashlib.sha256()
                        with open(submission_dst, "rb") as f:
                            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                                h.update(chunk)
                        node.submission_csv_sha256 = h.hexdigest()
                        logger.info(f"Saved submission.csv for node {node.step} to {submission_dst}")
                    except Exception as e:
                        logger.warning(f"Failed to set submission metadata for node {node.step}: {e}")
