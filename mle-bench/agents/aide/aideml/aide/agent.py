import shutil
import logging
import random
import time
from typing import Any, Callable, cast

import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .mcts_backtrack import create_selector
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")
#EDA_MEMORY = """Design: I will conduct exploratory data analysis on the dataset.
#Results: The relevant results of the exploratory data analysis mention that the target variable y is highly imbalanced with 87.93% being 0s and 12.06% being 1s. Two variables, pdays and poutcome, as well as poutcome and previous have very strong negative correlations while pdays and previous have a strong positive correlation. Other correlations are present as well: duration has the strongest positive correlation with the target.
#Validation Metric: N/A (EDA only)

#-------------------------------
EDA_MEMORY =""


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
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "has_csv_submission": {
                "type": "boolean",
                "description": "true if the code saves the predictions on the test data"
                " in a `submission.csv` file in the `./submission/` directory, otherwise false."
                " Note that the file MUST be saved in the ./submission/ directory for this to be evaluated as true."
                " Otherwise, it should be evaluated as false."
                " You can assume the ./submission/ directory exists and is writable.",
            },
            "summary": {
                "type": "string",
                "description": "write a short summary (2-3 sentences) describing "
                " the empirical findings. Alternatively mention if there is a bug or"
                " the submission.csv was not properly produced."
                " DO NOT suggest fixes or improvements.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": [
            "is_bug",
            "has_csv_submission",
            "summary",
            "metric",
            "lower_is_better",
        ],
    },
    description="Submit a review evaluating the output of the training script.",
)

error_comparison_func_spec = FunctionSpec(
    name="compare_errors",
    json_schema={
        "type": "object",
        "properties": {
            "is_same_error": {
                "type": "boolean",
                "description": "True if both errors represent the same underlying issue "
                "(same root cause, same type of bug), even if the exact message differs. "
                "False if they are fundamentally different errors.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why the errors are or aren't the same.",
            },
        },
        "required": ["is_same_error", "reasoning"],
    },
    description="Compare two error messages to determine if they represent the same underlying bug.",
)


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
        self.start_time = time.time()
        self.current_step = 0
        # Cache for LLM-based error comparisons to avoid repeated calls
        self._error_comparison_cache: dict[tuple[str, str], bool] = {}

        # Initialize MCTS node selector for intelligent backtracking
        mcts_enabled = getattr(self.acfg.search, 'use_mcts', False)
        if mcts_enabled:
            mcts_method = getattr(self.acfg.search, 'mcts_method', 'thompson')
            mcts_kwargs = {}

            if mcts_method == 'thompson':
                mcts_kwargs['prior_alpha'] = getattr(self.acfg.search, 'thompson_prior_alpha', 1.0)
                mcts_kwargs['prior_beta'] = getattr(self.acfg.search, 'thompson_prior_beta', 1.0)
            elif mcts_method == 'ucb1':
                mcts_kwargs['exploration_weight'] = getattr(self.acfg.search, 'ucb_exploration_weight', 1.414)

            self.node_selector = create_selector(mcts_method, **mcts_kwargs)
            logger.info(f"[MCTS ENABLED] Using {mcts_method} for node selection")
        else:
            self.node_selector = None
            logger.info("[MCTS DISABLED] Using random node selection")

        # Log error backtracking configuration
        # 0 or None = disabled, any positive integer = enabled with that threshold
        error_threshold = getattr(self.acfg.search, 'error_backtrack_threshold', 0)
        if error_threshold and error_threshold > 0:
            logger.info(f"[ERROR BACKTRACKING ENABLED] threshold={error_threshold} - Will backtrack after seeing same error {error_threshold} times")
        else:
            logger.info("[ERROR BACKTRACKING DISABLED] error_backtrack_threshold=0 or not set")

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.info("[search policy] drafting new node (not enough drafts)")
            return None

        # Get error backtrack threshold from config (0 = disabled)
        error_backtrack_threshold = getattr(search_cfg, 'error_backtrack_threshold', 0)

        # debugging with error-aware backtracking
        if random.random() < search_cfg.debug_prob:
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]

            if debuggable_nodes:
                node_to_debug = random.choice(debuggable_nodes)

                # Check for repeated errors using LLM - trigger backtracking (if enabled)
                if error_backtrack_threshold > 0:
                    error_count = self.count_same_errors_in_ancestors(node_to_debug)
                else:
                    error_count = 0  # Skip checking if backtracking disabled

                if error_backtrack_threshold > 0 and error_count >= error_backtrack_threshold:
                    logger.info(
                        f"[search policy] Same error seen {error_count + 1} times, triggering backtrack"
                    )

                    # Find where error first occurred
                    first_error_node = self.find_first_equivalent_error(node_to_debug)

                    if first_error_node and first_error_node.parent:
                        # Try to find an unexplored sibling at that level
                        siblings = self.journal.get_unexplored_siblings(first_error_node)

                        # Filter to good (non-buggy) siblings for improvement
                        good_siblings = [s for s in siblings if not s.is_buggy]

                        if good_siblings:
                            # Select a sibling to improve instead (using MCTS if enabled)
                            if self.node_selector:
                                backtrack_node = self.node_selector.select_best_node(good_siblings)
                                logger.info(
                                    f"[search policy] MCTS backtracking to sibling node {backtrack_node.id}"
                                )
                            else:
                                backtrack_node = random.choice(good_siblings)
                                logger.info(
                                    f"[search policy] Random backtracking to sibling node {backtrack_node.id}"
                                )
                            return backtrack_node

                        # If no good siblings, try buggy siblings with different errors
                        different_error_siblings = [
                            s for s in siblings
                            if s.is_buggy and not self.are_errors_equivalent(
                                node_to_debug.term_out, s.term_out
                            )
                        ]
                        if different_error_siblings:
                            if self.node_selector:
                                backtrack_node = self.node_selector.select_best_node(different_error_siblings)
                                logger.info(
                                    f"[search policy] MCTS backtracking to debug different error at node {backtrack_node.id}"
                                )
                            else:
                                backtrack_node = random.choice(different_error_siblings)
                                logger.info(
                                    f"[search policy] Random backtracking to debug different error at node {backtrack_node.id}"
                                )
                            return backtrack_node

                    # If backtracking failed, draft a new solution
                    logger.info("[search policy] Backtrack failed, drafting new node")
                    return None

                logger.info(f"[search policy] debugging node {node_to_debug.id}")
                return node_to_debug

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.info("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.info(f"[search policy] greedy node selected: node {greedy_node.id}")
        return greedy_node

    def are_errors_equivalent(self, error1: str, error2: str) -> bool:
        """Use LLM to determine if two errors are semantically the same (with caching)."""
        if not error1 or not error2:
            return False

        # Quick exact match check first (save LLM calls)
        if error1 == error2:
            return True

        # Truncate for comparison and caching
        error1_trunc = error1[:500]
        error2_trunc = error2[:500]

        # Check cache (order-independent key)
        cache_key = tuple(sorted([error1_trunc, error2_trunc]))
        if cache_key in self._error_comparison_cache:
            return self._error_comparison_cache[cache_key]

        prompt = {
            "Task": "Determine if these two error messages represent the same underlying bug or issue.",
            "Error 1": error1_trunc,
            "Error 2": error2_trunc,
            "Instructions": [
                "Consider the error TYPE (e.g., KeyError, ValueError, FileNotFoundError)",
                "Consider the ROOT CAUSE (e.g., missing column, wrong data type, file path issue)",
                "Ignore differences in line numbers, file paths, or variable values",
                "Return true if fixing one would likely fix the other",
            ],
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=error_comparison_func_spec,
                model=self.acfg.feedback.model,
                temperature=0.0,  # Deterministic for consistency
                convert_system_to_user=self.acfg.convert_system_to_user,
            ),
        )

        result = response["is_same_error"]
        logger.info(f"[error comparison] Same error: {result}, Reason: {response['reasoning']}")

        # Cache result
        self._error_comparison_cache[cache_key] = result
        return result

    def count_same_errors_in_ancestors(self, node: Node) -> int:
        """Count ancestors with semantically equivalent errors using LLM."""
        if not node.is_buggy or not node.term_out:
            return 0

        node_error = node.term_out
        count = 0

        for ancestor in node.get_ancestor_chain():
            if ancestor.is_buggy and ancestor.term_out:
                if self.are_errors_equivalent(node_error, ancestor.term_out):
                    count += 1

        return count

    def find_first_equivalent_error(self, node: Node) -> Node | None:
        """Find first ancestor with semantically equivalent error."""
        if not node.is_buggy or not node.term_out:
            return None

        node_error = node.term_out

        for ancestor in node.get_ancestor_chain():  # Root first
            if ancestor.is_buggy and ancestor.term_out:
                if self.are_errors_equivalent(node_error, ancestor.term_out):
                    return ancestor

        return None

    def _compute_reward(self, node: Node) -> float:
        """Compute normalized reward for MCTS update.

        Converts the node's metric into a reward value in [0, 1] for MCTS learning.

        Args:
            node: The node to compute reward for

        Returns:
            Reward in [0, 1]:
            - 0.0 for buggy nodes
            - Normalized metric value (0-1) for successful nodes
        """
        if node.is_buggy or node.metric is None:
            return 0.0

        # Get recent good metrics for normalization (last 20 for efficiency)
        recent_good = [
            n.metric for n in self.journal.nodes[-20:]
            if not n.is_buggy and n.metric is not None
        ]

        if not recent_good:
            return 0.5  # Neutral reward if no baseline

        metric_values = [m.value for m in recent_good]
        min_val, max_val = min(metric_values), max(metric_values)

        if max_val == min_val:
            return 0.5  # All metrics same, neutral reward

        # Normalize to [0, 1]
        normalized = (node.metric.value - min_val) / (max_val - min_val)

        # Flip if lower is better (e.g., MSE, cross-entropy)
        if not node.metric.maximize:
            normalized = 1.0 - normalized

        return normalized

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
        tot_time_elapsed = time.time() - self.start_time
        tot_time_remaining = self.acfg.time_limit - tot_time_elapsed
        exec_timeout = int(min(self.cfg.exec.timeout, tot_time_remaining))

        impl_guideline = [
            f"<TOTAL_TIME_REMAINING: {format_time(tot_time_remaining)}>",
            f"<TOTAL_STEPS_REMAINING: {self.acfg.steps - self.current_step}>",
            "The code should **implement the proposed solution**, **print the value of the evaluation metric computed on a hold-out validation set**,",
            "**AND MOST IMPORTANTLY SAVE PREDICTIONS ON THE PROVIDED UNLABELED TEST DATA IN A `submission.csv` FILE IN THE ./submission/ DIRECTORY.**",
            "The code should be a single-file python program that is self-contained and can be executed as-is.",
            "No parts of the code should be skipped, don't terminate the before finishing the script.",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(exec_timeout)}.",
            'All the provided input data is stored in "./input" directory.',
            '**You MUST submit predictions on the provided unlabeled test data in a `submission.csv` file** file in the "./working" directory as described in the task description** This is extremely important since this file is used for grading/evaluation. DO NOT FORGET THE submission.csv file!',
            'You can also use the "./working" directory to store any temporary files that your code needs to create.',
            "REMEMBER THE ./submission/submission.csv FILE!!!!! The correct directory is important too.",
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a predict() function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The prediction function should be well-documented, especially the function signature."
            )

        if self.acfg.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.acfg.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
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

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
                convert_system_to_user=self.acfg.convert_system_to_user,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            logger.info("Plan + code extraction failed, retrying...")
        logger.info("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
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
            "Prior Analysis": "Here are the relevant results of the EDA: the target variable y is highly imbalanced with 87.93% being 0s and 12.06% being 1s. Also consider the file: openhands/playground-series-s5e8/correlation_matrix.csv, which contains correlation information between the various variables. Notably, pdays and poutcome, as well as poutcome and previous have very strong negative correlations while pdays and previous have a strong positive correlation. Other correlations are present as well: duration has the strongest positive correlation with the target. Accordingly you are allowed to downsample the majority class, and upweight the downsampled class as needed.",
            "Memory": EDA_MEMORY + self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same modelling solution but keep the evaluation the same.",
                "The solution sketch should be 3-5 sentences.",
                "Propose an evaluation metric that is reasonable for this task.",
                #"Don't suggest to do EDA.",
                "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code)
        logger.info(f"Drafted new node {new_node.id}")
        return new_node

    def _improve(self, parent_node: Node) -> Node:
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
            "Memory": EDA_MEMORY + self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
                #"Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Improved node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def _debug(self, parent_node: Node) -> Node:
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
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                #"Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        new_node = Node(plan=plan, code=code, parent=parent_node)
        logger.info(f"Debugged node {parent_node.id} to create new node {new_node.id}")
        return new_node

    def update_data_preview(
        self,
    ):
        self.data_preview = data_preview.generate(self.cfg.workspace_dir)

    def step(self, exec_callback: ExecCallbackType):
        # clear the submission dir from previous steps
        shutil.rmtree(self.cfg.workspace_dir / "submission", ignore_errors=True)
        (self.cfg.workspace_dir / "submission").mkdir(exist_ok=True)

        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.info(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        result_node = self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        # handle final cases where we missed buggy nodes somehow
        if not result_node.is_buggy:
            if not (self.cfg.workspace_dir / "submission" / "submission.csv").exists():
                result_node.is_buggy = True
                result_node.metric = WorstMetricValue()
                logger.info(
                    f"Actually, node {result_node.id} did not produce a submission.csv"
                )
        self.journal.append(result_node)

        # Update MCTS node selector with reward (if enabled)
        if self.node_selector:
            reward = self._compute_reward(result_node)
            self.node_selector.update_reward(result_node, reward)
            logger.info(f"[MCTS] Updated node {result_node.id} with reward={reward:.3f}")

        # if the result_node is the best node, cache its submission.csv and solution.py
        # to best_solution/ by copying it there
        best_node = self.journal.get_best_node()
        if best_node is not None:
            if best_node.id == result_node.id:
                logger.info(f"Node {result_node.id} is the best node so far")
                best_solution_dir = self.cfg.workspace_dir / "best_solution"
                best_solution_dir.mkdir(exist_ok=True, parents=True)
                # copy submission/submission.csv to best_submission/submission.csv
                best_submission_dir = self.cfg.workspace_dir / "best_submission"
                best_submission_dir.mkdir(exist_ok=True, parents=True)
                shutil.copy(
                    self.cfg.workspace_dir / "submission" / "submission.csv",
                    best_submission_dir,
                )
                # copy solution.py and relevant node id to best_solution/
                with open(best_solution_dir / "solution.py", "w") as f:
                    f.write(result_node.code)
                # take note of the node id of the best node
                with open(best_solution_dir / "node_id.txt", "w") as f:
                    f.write(str(result_node.id))
            else:
                logger.info(f"Node {result_node.id} is not the best node")
                logger.info(f"Node {best_node.id} is still the best node")
        self.current_step += 1

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult) -> Node:
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
        prompt = {
            "Introduction": introduction,
            "Task description": self.task_desc,
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
            ),
        )

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        # do an extra check, to catch cases where judge fails
        has_csv_submission = (
            self.cfg.workspace_dir / "submission" / "submission.csv"
        ).exists()

        node.analysis = response["summary"]
        node.is_buggy = (
            response["is_bug"]
            or node.exc_type is not None
            or response["metric"] is None
            or response["has_csv_submission"] == False
            or has_csv_submission == False
        )

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

        return node
