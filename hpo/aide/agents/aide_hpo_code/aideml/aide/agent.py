import shutil
import logging
import random
import time
from typing import Any, Callable, cast


import humanize
from .backend import FunctionSpec, compile_prompt_to_md, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code


logger = logging.getLogger("aide")


hpo_scoring_func_spec = FunctionSpec(
   name="score_hyperparameter_tuning",
   json_schema={
       "type": "object",
       "properties": {
           "score": {
               "type": "integer",
               "description": "Score the quality of hyperparameter tuning on a 0-3 scale: "
               "0 = none (no hyperparameter tuning), "
               "1 = superficial (minimal tuning, e.g., only 2-3 values tested), "
               "2 = moderate (reasonable tuning with multiple hyperparameters or systematic search), "
               "3 = extensive (comprehensive tuning with multiple hyperparameters, systematic search, and proper validation).",
               "minimum": 0,
               "maximum": 3,
           },
       },
       "required": ["score"],
   },
   description="Score the quality of hyperparameter tuning in the code.",
)




def _score_hyperparameter_tuning(code: str) -> int:
   """
   Evaluates the quality of hyperparameter tuning on a 0-3 scale using LLM.
  
   Returns:
       0 = none (no hyperparameter tuning)
       1 = superficial (minimal tuning)
       2 = moderate (reasonable tuning)
       3 = extensive (comprehensive tuning)
   """
   system_prompt = (
       "You are a strict ML code reviewer evaluating hyperparameter tuning quality.\n"
       "Score the hyperparameter tuning in the following Python code on a 0-3 scale:\n\n"
       "0 = none: No hyperparameter tuning (only fixed/default hyperparameters)\n"
       "1 = superficial: Minimal tuning (e.g., only 2-3 values tested for 1 hyperparameter, "
       "or very small grid/random search with <5 iterations)\n"
       "2 = moderate: Reasonable tuning (multiple hyperparameters tested, systematic search "
       "with ≥5 iterations, proper validation)\n"
       "3 = extensive: Comprehensive tuning (multiple hyperparameters, systematic search "
       "with ≥10 iterations, proper validation, best params reused for final training)\n\n"
       "DO NOT count:\n"
       "- cross_val_score alone without hyperparameter search\n"
       "- fixed hyperparameters\n"
       "- train/val splits without search\n"
       "- hyperparameter tuning that is not used in final model training\n\n"
       "Respond with ONLY the integer score (0, 1, 2, or 3)."
   )


   user_prompt = f"```python\n{code}\n```"


   try:
       resp = query(
           system_message=system_prompt,
           user_message=user_prompt,
           func_spec=hpo_scoring_func_spec,
           model="gpt-4o-2024-08-06",
           temperature=0,
           convert_system_to_user=False,
       )
       score = int(resp.get("score", 0))
       return max(0, min(3, score))  # Clamp to [0, 3]
   except Exception as e:
       logger.warning(f"HPO scoring failed: {e}, defaulting to 0")
       return 0


def format_time(time_in_sec: int):
   return f"{time_in_sec // 3600}hrs {(time_in_sec % 3600) // 60}mins {time_in_sec % 60}secs"




def _apply_structural_hpo_caps(code: str, hpo_score: int) -> int:
   """
   Apply structural caps to prevent fake/weak HPO.
   Returns the capped HPO score.
   """
   import re
  
   # Cap RandomizedSearchCV with n_iter < 5
   randomized_search_pattern = r"RandomizedSearchCV\s*\([^)]*n_iter\s*=\s*(\d+)"
   match = re.search(randomized_search_pattern, code, re.IGNORECASE)
   if match:
       n_iter = int(match.group(1))
       if n_iter < 5:
           hpo_score = min(hpo_score, 1)
           logger.info(f"Capped HPO score to 1 due to RandomizedSearchCV with n_iter={n_iter} < 5")
  
   # Cap GridSearchCV with very small grids (check for parameter grids with < 3 values)
   grid_search_pattern = r"GridSearchCV\s*\([^)]*param_grid\s*=\s*\{[^}]*\}"
   if re.search(grid_search_pattern, code, re.IGNORECASE):
       # Check if param_grid has very few values (simple heuristic: count colons in dict)
       param_grid_pattern = r"param_grid\s*=\s*\{([^}]+)\}"
       match = re.search(param_grid_pattern, code, re.IGNORECASE)
       if match:
           param_grid_content = match.group(1)
           # Count list lengths (rough heuristic)
           list_counts = len(re.findall(r"\[[^\]]+\]", param_grid_content))
           if list_counts < 2 or (list_counts == 2 and all(len(re.findall(r"\[([^\]]+)\]", part)) < 3 for part in param_grid_content.split(","))):
               hpo_score = min(hpo_score, 1)
               logger.info(f"Capped HPO score to 1 due to very small grid search")
  
   # Check if best parameters are never reused for final training
   # Look for pattern: search.fit() but no model with best_params_ or best_estimator_
   has_search = re.search(r"(GridSearchCV|RandomizedSearchCV|Optuna|Hyperopt)", code, re.IGNORECASE)
   if has_search:
       # Check if best params are used in final training
       has_best_params_usage = re.search(
           r"(best_params_|best_estimator_|\.best_estimator|\.best_params)", code, re.IGNORECASE
       )
       if not has_best_params_usage:
           hpo_score = min(hpo_score, 1)
           logger.info(f"Capped HPO score to 1: best parameters not reused for final training")
  
   return hpo_score




def _check_model_family_diversity(code: str) -> float:
   """
   Check model family diversity. Returns reward:
   +0.1 for single focused model family
   -0.1 for multiple model families/ensembles early
   """
   import re
  
   model_families = {
       "tree": ["RandomForest", "DecisionTree", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "GradientBoosting"],
       "linear": ["LinearRegression", "LogisticRegression", "Ridge", "Lasso", "ElasticNet", "SGD"],
       "neural": ["MLP", "NeuralNetwork", "Sequential", "torch.nn", "tf.keras"],
       "svm": ["SVC", "SVR", "SVM"],
       "naive_bayes": ["GaussianNB", "MultinomialNB", "BernoulliNB"],
   }
  
   found_families = set()
   for family, models in model_families.items():
       for model in models:
           if re.search(rf"\b{model}\b", code, re.IGNORECASE):
               found_families.add(family)
               break
  
   # Check for ensemble patterns
   has_ensemble = re.search(r"(Voting|Stacking|Blending|ensemble)", code, re.IGNORECASE)
  
   if len(found_families) == 1 and not has_ensemble:
       return 0.1  # Single focused model family
   elif len(found_families) > 1 or has_ensemble:
       return -0.1  # Multiple families or ensemble
   return 0.0




def _check_hpo_correctness(code: str, term_out: str) -> float:
   """
   Check if HPO is used correctly. Returns penalty if HPO is performed incorrectly.
   """
   import re
  
   has_hpo = re.search(r"(GridSearchCV|RandomizedSearchCV|Optuna|Hyperopt|hyperparameter)", code, re.IGNORECASE)
   if not has_hpo:
       return 0.0
  
   penalty = 0.0
  
   # Check if validation improves vs baseline (heuristic: look for metric improvements in output)
   # This is a simple check - in practice, this would need more sophisticated analysis
   if "best_score" in term_out.lower() or "best_params" in term_out.lower():
       # If best params are found, check if they're used
       if not re.search(r"(best_params_|best_estimator_|\.best_estimator)", code, re.IGNORECASE):
           penalty -= 0.05
           logger.info("HPO correctness penalty: best hyperparameters found but not reused")
  
   return penalty




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


   def search_policy(self) -> Node | None:
       """Select a node to work on (or None to draft a new node)."""
       search_cfg = self.acfg.search


       # initial drafting
       if len(self.journal.draft_nodes) < search_cfg.num_drafts:
           logger.info("[search policy] drafting new node (not enough drafts)")
           return None


       # debugging
       if random.random() < search_cfg.debug_prob:
           # nodes that are buggy + leaf nodes + debug depth < max debug depth
           debuggable_nodes = [
               n
               for n in self.journal.buggy_nodes
               if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
           ]
           if debuggable_nodes:
               node_to_debug = random.choice(debuggable_nodes)
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
           "Memory": self.journal.generate_summary(),
           "Instructions": {},
       }
       prompt["Instructions"] |= self._prompt_resp_fmt
       # Build solution sketch guidelines
       solution_guidelines = [
           "This first solution design should be relatively simple, without ensembling.",
           "Take the Memory section into consideration when proposing the design,"
           " don't propose the same modelling solution but keep the evaluation the same.",
           "The solution sketch should be 3-5 sentences.",
           "Propose an evaluation metric that is reasonable for this task.",
           "Don't suggest to do EDA.",
           "The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
       ]
      
       # Add HPO encouragement after 30% of steps
       progress_ratio = self.current_step / max(self.acfg.steps, 1)
       if progress_ratio > 0.3:
           solution_guidelines.insert(1,
               "**Note: Since you're past the initial exploration phase, consider including basic hyperparameter tuning** "
               "(e.g., testing 3-5 values for key hyperparameters like learning_rate, n_estimators, or regularization strength) "
               "to establish a stronger baseline."
           )
       else:
           solution_guidelines.insert(1, "You may include basic hyperparameter tuning if appropriate.")
      
       prompt["Instructions"] |= {
           "Solution sketch guideline": solution_guidelines,
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
           "Memory": self.journal.generate_summary(),
           "Instructions": {},
       }
       prompt["Previous solution"] = {
           "Code": wrap_code(parent_node.code),
       }


       prompt["Instructions"] |= self._prompt_resp_fmt
      
       # Check HPO status and add guidance
       parent_hpo_score = getattr(parent_node, 'hpo_score', 0)
       if parent_hpo_score == 0:
           hpo_guideline = [
               "**CRITICAL: The previous solution has NO hyperparameter tuning (HPO score: 0). "
               "You MUST add hyperparameter tuning** using GridSearchCV, RandomizedSearchCV, Optuna, "
               "Hyperopt, or manual loops searching over ≥3 values for ≥1 hyperparameter. "
               "This is required to improve model performance."
           ]
       elif parent_hpo_score == 1:
           hpo_guideline = [
               "**IMPORTANT: The previous solution has only superficial hyperparameter tuning (HPO score: 1). "
               "You should improve it** by testing more hyperparameter values (≥5 iterations), "
               "searching over multiple hyperparameters, or using a more systematic approach."
           ]
       elif parent_hpo_score == 2:
           hpo_guideline = [
               "The previous solution has moderate hyperparameter tuning (HPO score: 2). "
               "Consider expanding the search space or tuning additional hyperparameters to reach extensive tuning (score 3)."
           ]
       else:
           hpo_guideline = []
      
       prompt["Instructions"] |= {
           "Solution improvement sketch guideline": [
               "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
               "You should be very specific and should only propose a single actionable improvement.",
               "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
               "Take the Memory section into consideration when proposing the improvement.",
               *hpo_guideline,
               "The solution sketch should be 3-5 sentences.",
               "Don't suggest to do EDA.",
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
       bugfix_guidelines = [
           "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
           "Don't suggest to do EDA.",
       ]
      
       # Check HPO status and add guidance if missing
       parent_hpo_score = getattr(parent_node, 'hpo_score', 0)
       if parent_hpo_score == 0:
           bugfix_guidelines.insert(1,
               "**IMPORTANT: The previous solution has NO hyperparameter tuning. "
               "After fixing the bug, you should also add hyperparameter tuning** "
               "(e.g., GridSearchCV, RandomizedSearchCV, Optuna, Hyperopt, or manual loops "
               "searching over ≥3 values for ≥1 hyperparameter) to improve performance."
           )
      
       prompt["Instructions"] |= {
           "Bugfix improvement sketch guideline": bugfix_guidelines,
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


       logger.info(f"Scoring hyperparameter tuning for node {node.id}")
       hpo_score = _score_hyperparameter_tuning(node.code)
       logger.info(f"Node {node.id} initial HPO score: {hpo_score}")
      
       # Apply structural caps to prevent fake/weak HPO
       hpo_score = _apply_structural_hpo_caps(node.code, hpo_score)
       logger.info(f"Node {node.id} HPO score after structural caps: {hpo_score}")
      
       # Store HPO score on node
       node.hpo_score = hpo_score


       if node.is_buggy:
           logger.info(
               f"Parsed results: Node {node.id} is buggy and/or did not produce a submission.csv"
           )
           node.metric = WorstMetricValue()
       else:
           logger.info(f"Parsed results: Node {node.id} is not buggy")
           base_metric = MetricValue(
               response["metric"], maximize=not response["lower_is_better"]
           )
          
           # Apply soft HPO reward shaping (curriculum-based)
           progress_ratio = self.current_step / max(self.acfg.steps, 1)
           hpo_reward = self._get_hpo_reward(hpo_score, progress_ratio)
          
           # Model family diversity reward
           diversity_reward = _check_model_family_diversity(node.code)
          
           # HPO correctness penalty
           correctness_penalty = _check_hpo_correctness(node.code, node.term_out)
          
           # Apply rewards/penalties to metric value
           # We adjust the metric value by a small amount based on HPO quality
           if base_metric.value is not None:
               total_adjustment = hpo_reward + diversity_reward + correctness_penalty
               # Scale adjustment based on metric magnitude (avoid overwhelming small metrics)
               metric_scale = abs(base_metric.value) if abs(base_metric.value) > 0.01 else 1.0
               adjusted_value = base_metric.value + (total_adjustment * metric_scale * 0.1)
              
               # Create adjusted metric (but keep original for logging)
               node.metric = MetricValue(
                   adjusted_value, maximize=base_metric.maximize
               )
               logger.info(
                   f"Node {node.id} metric adjusted: base={base_metric.value:.6f}, "
                   f"hpo_reward={hpo_reward:.3f}, diversity={diversity_reward:.3f}, "
                   f"correctness={correctness_penalty:.3f}, final={adjusted_value:.6f}"
               )
           else:
               node.metric = base_metric


       return node
  
   def _get_hpo_reward(self, hpo_score: int, progress_ratio: float) -> float:
       """
       Get HPO reward based on score and search progress (curriculum-style).
      
       Early search (0-40%): allow no/weak HPO, only apply penalties
       Mid search (40-70%): reward score ≥2
       Late search (70-100%): strongly reward score 3
       """
       if progress_ratio < 0.4:  # Early search
           if hpo_score == 0:
               return -0.3
           elif hpo_score == 1:
               return -0.1
           else:
               return 0.0  # No penalty for moderate/extensive HPO early
       elif progress_ratio < 0.7:  # Mid search
           if hpo_score == 0:
               return -0.3
           elif hpo_score == 1:
               return -0.1
           elif hpo_score >= 2:
               return 0.15
           return 0.0
       else:  # Late search
           if hpo_score == 0:
               return -0.3
           elif hpo_score == 1:
               return -0.1
           elif hpo_score == 2:
               return 0.15
           elif hpo_score == 3:
               return 0.35
           return 0.0

