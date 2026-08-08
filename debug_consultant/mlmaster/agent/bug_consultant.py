"""
Debug Consultant: Memory Writer + Retrieval-Only Reader for Debugging History
(Adapted for ML-Master from the AIDE implementation)

Design goals (ICML submission aligned):
- In-context RL (no parameter updates): improve behavior via better context and action selection.
- World model writing: persist compact, policy-relevant "what works / what fails" rules (not raw logs).
- Separation of retrieval and execution: provide curated context to the code-generating actor.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from backend import FunctionSpec, query

if TYPE_CHECKING:
    from search.node import Node
    from search.journal import Journal
    from utils.config_mcts import Config

logger = logging.getLogger("ml-master")


# ═══════════════════════════════════════════════════════════
# Data Structures (Simple Storage)
# ═══════════════════════════════════════════════════════════


@dataclass
class DebugTrial:
    """Record of a single debug attempt on a parent bug."""

    attempt_num: int
    node_step: int
    debug_plan: str
    code: str
    outcome: str  # "success" | "failed"
    error_type: Optional[str] = None
    error_output: Optional[str] = None
    why_worked: Optional[str] = None
    why_failed: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class BugRecord:
    """Organized record of a bug and all attempts to fix it."""

    bug_id: str
    original_node_step: int
    error_type: str
    buggy_code: str
    buggy_output: str
    original_plan: str

    trials: list[DebugTrial] = field(default_factory=list)
    final_outcome: str = "in_progress"  # "success" | "abandoned" | "in_progress" | "dead"
    is_dead: bool = False  # True for timeout/unfixable bugs - never debug these

    # LLM-summarized fields at bug start (for RAG)
    error_signature: Optional[str] = None
    error_category: Optional[str] = None
    initial_hypothesis: Optional[str] = None
    context_tags: list[str] = field(default_factory=list)

    # Extracted / summarized fields (from trials and completion)
    root_cause: Optional[str] = None
    successful_strategy: Optional[str] = None
    failed_strategies: list[str] = field(default_factory=list)
    learned_constraints: list[str] = field(default_factory=list)
    lesson: Optional[str] = None

    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


# ═══════════════════════════════════════════════════════════
# Function Specs for LLM Calls
# ═══════════════════════════════════════════════════════════


retrieve_spec = FunctionSpec(
    name="retrieve_optimal_context",
    json_schema={
        "type": "object",
        "properties": {
            "selected_bug_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Bug IDs to retrieve - choose an optimal number (not fixed).",
            },
            "reasoning": {
                "type": "string",
                "description": "Why these bugs? Why this number?",
            },
            "key_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key patterns across selected bugs",
            },
        },
        "required": ["selected_bug_ids", "reasoning", "key_patterns"],
    },
    description="Select the most relevant historical bugs for the current issue and explain the reasoning and shared patterns.",
)

summarize_bug_record_spec = FunctionSpec(
    name="summarize_bug_record",
    json_schema={
        "type": "object",
        "properties": {
            "root_cause": {
                "type": "string",
                "description": "The ACTUAL root cause explanation - what technically caused this bug. Be specific and concise (1-2 sentences).",
            },
            "successful_strategy": {
                "type": "string",
                "description": "Specific strategy that worked, in one concise sentence.",
            },
            "failed_strategies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific strategies that failed with WHY each failed.",
            },
            "lesson": {
                "type": "string",
                "description": "Actionable, reusable pattern that can prevent this exact mistake in future.",
            },
        },
        "required": ["root_cause", "lesson"],
    },
    description="Extract root cause, strategies, and reusable lessons from a completed bug record.",
)

summarize_bug_start_spec = FunctionSpec(
    name="summarize_bug_start",
    json_schema={
        "type": "object",
        "properties": {
            "error_signature": {
                "type": "string",
                "description": "EXACT error message from output.",
            },
            "error_category": {
                "type": "string",
                "description": "Category: TYPE_ERROR, VALUE_ERROR, ATTRIBUTE_ERROR, KEY_ERROR, INDEX_ERROR, IMPORT_ERROR, FILE_NOT_FOUND, TIMEOUT, UNKNOWN, or OTHER",
            },
            "context_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Library/function names from the error.",
            },
            "is_unfixable": {
                "type": "boolean",
                "description": "True ONLY for actual timeout or OOM. False for all code errors.",
            },
        },
        "required": ["error_signature", "error_category", "context_tags", "is_unfixable"],
    },
    description="Extract error info from output. Only include what's actually in the error - no speculation.",
)

summarize_trial_failure_spec = FunctionSpec(
    name="summarize_trial_failure",
    json_schema={
        "type": "object",
        "properties": {
            "what_was_tried": {
                "type": "string",
                "description": "Brief description of what approach was attempted.",
            },
            "actual_error": {
                "type": "string",
                "description": "The EXACT error message from output.",
            },
            "failed_strategy_summary": {
                "type": "string",
                "description": "Format: '<what was tried> → <actual error>'.",
            },
        },
        "required": ["what_was_tried", "actual_error", "failed_strategy_summary"],
    },
    description="Summarize what was tried and what error occurred.",
)

summarize_trial_success_spec = FunctionSpec(
    name="summarize_trial_success",
    json_schema={
        "type": "object",
        "properties": {
            "why_worked": {
                "type": "string",
                "description": "Specific reason WHY this approach succeeded.",
            },
            "successful_strategy_summary": {
                "type": "string",
                "description": "One-line summary of successful strategy.",
            },
            "key_insight": {
                "type": "string",
                "description": "Key insight that made this work.",
            },
        },
        "required": ["why_worked", "successful_strategy_summary", "key_insight"],
    },
    description="Summarize a successful debug trial for RL.",
)

extract_error_signature_spec = FunctionSpec(
    name="extract_error_signature",
    json_schema={
        "type": "object",
        "properties": {
            "error_signature": {
                "type": "string",
                "description": "Compact, specific error signature.",
            },
            "error_category": {
                "type": "string",
                "description": "High-level category: API_MISUSE, MISSING_DATA, TYPE_ERROR, IMPORT_ERROR, LOGIC_ERROR, VALUE_ERROR, ATTRIBUTE_ERROR, KEY_ERROR, INDEX_ERROR, or OTHER",
            },
        },
        "required": ["error_signature", "error_category"],
    },
    description="Extract a meaningful error signature and category from execution output for bug tracking.",
)

distill_world_model_spec = FunctionSpec(
    name="distill_world_model",
    json_schema={
        "type": "object",
        "properties": {
            "patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "The banned pattern as text description.",
                        },
                        "error_type": {
                            "type": "string",
                            "description": "The error type.",
                        },
                        "fix_syntax": {
                            "type": "string",
                            "description": "Proven fix as SHORT code syntax (ONLY from Proven Successes list).",
                        },
                    },
                    "required": ["pattern", "error_type"],
                },
                "description": "List of banned code patterns that cause crashes",
            },
        },
        "required": ["patterns"],
    },
    description="Extract banned code patterns from crash errors",
)

# ═══════════════════════════════════════════════════════════
# Bug Consultant Class
# ═══════════════════════════════════════════════════════════


class BugConsultant:
    """
    Memory writer + retrieval-only reader for debugging history.
    Adapted for ML-Master (query() requires cfg parameter).
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 1,
        save_dir: Optional[Path] = None,
        cfg: Optional["Config"] = None,
        max_bug_records: int = 50,
        advice_budget_chars: int = 200000,
        max_active_bugs: int = 200,
        max_trials_per_bug: int = 20,
        delete_pruned_bug_files: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.cfg = cfg
        self.max_bug_records = max_bug_records
        self.advice_budget_chars = int(advice_budget_chars)
        self.max_active_bugs = max_active_bugs
        self.max_trials_per_bug = max_trials_per_bug
        self.delete_pruned_bug_files = delete_pruned_bug_files

        self.save_dir = save_dir
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.bug_records: dict[str, BugRecord] = {}
        self.active_bugs: dict[str, BugRecord] = {}

        # World model version counter
        self.world_model_version: int = 0

        # Distilled guidance cache
        self._distilled_guidance: str = ""
        self._distilled_content_hash: str = ""
        self._distilled_version: int = -1
        self._guidance_dirty: bool = False
        self._guidance_lock = threading.Lock()

        logger.info("Debug consultant initialized (model: %s)", model)

    def _query(self, **kwargs):
        """Wrapper around backend.query that injects cfg."""
        return query(cfg=self.cfg, **kwargs)

    # ═══════════════════════════════════════════════════════════
    # Writer: Bug record lifecycle
    # ═══════════════════════════════════════════════════════════

    def start_bug_record(self, node: "Node") -> str:
        if node.step is None:
            raise ValueError("Cannot start bug record: node.step is None")

        self._enforce_active_bug_limit()

        bug_id = f"bug_{node.step}"
        full_output = "".join(node._term_out) if hasattr(node, "_term_out") and node._term_out else (node.term_out or "")
        record = BugRecord(
            bug_id=bug_id,
            original_node_step=node.step,
            error_type=node.exc_type or "Unknown",
            buggy_code=node.code,
            buggy_output=full_output,
            original_plan=node.plan or "",
        )

        logger.debug("Summarizing bug start for %s", bug_id)
        summary = self._summarize_bug_start(record)
        record.error_signature = summary.get("error_signature")
        record.error_category = summary.get("error_category")
        record.initial_hypothesis = summary.get("initial_hypothesis")
        record.context_tags = summary.get("context_tags", [])

        if summary.get("is_unfixable", False):
            record.is_dead = True
            record.final_outcome = "dead"
            record.lesson = f"Unfixable error: {record.initial_hypothesis}"
            self.bug_records[bug_id] = record
            logger.info("Created DEAD bug record: %s (unfixable: %s)", bug_id, record.error_category)
            self._save_bug_record(record, active=False)
            self._save_world_model()
            return ""

        self.active_bugs[bug_id] = record
        logger.info("Started bug record: %s (category: %s)", bug_id, record.error_category)
        self._save_bug_record(record, active=True)
        return bug_id

    def record_trial(
        self,
        bug_id: str,
        node: "Node",
        outcome: str,
        why_worked: Optional[str] = None,
        why_failed: Optional[str] = None,
    ) -> None:
        if bug_id not in self.active_bugs:
            logger.warning("Bug %s not found in active bugs", bug_id)
            return
        if node.step is None:
            logger.warning("Skipping trial record: node.step is None")
            return

        record = self.active_bugs[bug_id]
        full_output = None
        if outcome == "failed":
            full_output = "".join(node._term_out) if hasattr(node, "_term_out") and node._term_out else (node.term_out or "")
        trial = DebugTrial(
            attempt_num=len(record.trials) + 1,
            node_step=node.step,
            debug_plan=node.plan or "",
            code=node.code,
            outcome=outcome,
            error_type=node.exc_type if outcome == "failed" else None,
            error_output=full_output,
            why_worked=why_worked,
            why_failed=why_failed,
        )
        record.trials.append(trial)

        if outcome == "failed":
            logger.debug("Summarizing failed trial #%s for %s", trial.attempt_num, bug_id)
            summary = self._summarize_trial_failure(trial, record)
            trial.why_failed = summary.get("actual_error", trial.why_failed)
            failed_strategy = summary.get("failed_strategy_summary", "")
            if failed_strategy and failed_strategy not in record.failed_strategies:
                record.failed_strategies.append(failed_strategy)
            logger.info("Recorded FAILED trial #%s for %s: %s", trial.attempt_num, bug_id, failed_strategy)

        elif outcome == "success":
            logger.debug("Summarizing successful trial #%s for %s", trial.attempt_num, bug_id)
            summary = self._summarize_trial_success(trial, record)
            trial.why_worked = summary.get("why_worked", trial.why_worked)
            successful_strategy = summary.get("successful_strategy_summary", "")
            if successful_strategy:
                record.successful_strategy = successful_strategy
            logger.info("Recorded SUCCESS trial #%s for %s: %s", trial.attempt_num, bug_id, successful_strategy)

        self._save_bug_record(record, active=True)

        if self.max_trials_per_bug > 0 and len(record.trials) >= self.max_trials_per_bug:
            self._auto_abandon_bug_record(
                bug_id,
                reason=f"Exceeded max_trials_per_bug={self.max_trials_per_bug}",
            )

    def complete_bug_record(
        self, bug_id: str, outcome: str, *, journal: Optional["Journal"] = None
    ) -> None:
        if bug_id not in self.active_bugs:
            return

        record = self.active_bugs[bug_id]
        record.final_outcome = outcome

        summary = self._summarize_bug_record(record)
        record.root_cause = summary.get("root_cause") or record.root_cause
        record.successful_strategy = summary.get("successful_strategy") or record.successful_strategy
        record.failed_strategies = summary.get("failed_strategies") or record.failed_strategies
        record.lesson = summary.get("lesson") or record.lesson

        self.bug_records[bug_id] = record
        self.active_bugs.pop(bug_id, None)
        self._prune_bug_records()

        logger.info("Completed bug record %s", bug_id)
        self._save_bug_record(record, active=False)
        self._save_world_model(journal=journal)

    def abandon_bug_record(self, bug_id: str, *, journal: Optional["Journal"] = None) -> None:
        self.complete_bug_record(bug_id, outcome="abandoned", journal=journal)

    # ═══════════════════════════════════════════════════════════
    # Writer: Backward-compatible ingestion API
    # ═══════════════════════════════════════════════════════════

    def learn_from_bug(self, node: "Node", journal: Optional["Journal"] = None) -> None:
        if node.step is None:
            logger.warning("Skipping BugConsultant update: node.step is None")
            return

        if node.parent is not None and getattr(node.parent, "is_buggy", False):
            parent_step = getattr(node.parent, "step", None)
            if parent_step is None:
                return
            parent_bug_id = f"bug_{parent_step}"
            if parent_bug_id not in self.active_bugs:
                try:
                    self.start_bug_record(node.parent)
                except Exception:
                    return

            if node.is_buggy:
                self.record_trial(
                    parent_bug_id,
                    node,
                    outcome="failed",
                    why_failed=node.analysis or (f"Still failing: {node.exc_type}" if node.exc_type else "Still failing"),
                )
                self._guidance_dirty = True
                self._save_world_model(journal=journal)
                return

            self.record_trial(
                parent_bug_id,
                node,
                outcome="success",
                why_worked=node.analysis or "Executed successfully",
            )
            self.complete_bug_record(parent_bug_id, outcome="success", journal=journal)
            self._guidance_dirty = True
            return

        if node.is_buggy:
            self.start_bug_record(node)
            self._guidance_dirty = True
            self._save_world_model(journal=journal)
            return

        # Non-bug node, no buggy parent: no bug state changed, don't dirty guidance

    # ═══════════════════════════════════════════════════════════
    # Reader: Retrieval and formatting
    # ═══════════════════════════════════════════════════════════

    def retrieve_relevant_context(
        self,
        current_error_type: str,
        current_error_msg: str,
        current_code: str,
        original_plan: str,
    ) -> dict:
        if not self.bug_records and not self.active_bugs:
            return {"selected_bugs": [], "reasoning": "No historical bugs yet", "key_patterns": []}

        all_bugs = []

        for bug_id, record in self.bug_records.items():
            all_bugs.append({
                "bug_id": bug_id, "status": "completed",
                "error_type": record.error_type,
                "error_signature": record.error_signature,
                "error_category": record.error_category,
                "context_tags": record.context_tags,
                "initial_hypothesis": record.initial_hypothesis,
                "successful_strategy": record.successful_strategy,
                "failed_strategies": record.failed_strategies,
                "learned_constraints": record.learned_constraints,
                "root_cause": record.root_cause,
                "lesson": record.lesson,
                "outcome": record.final_outcome,
                "trials_count": len(record.trials),
            })

        for bug_id, record in self.active_bugs.items():
            all_bugs.append({
                "bug_id": bug_id, "status": "active",
                "error_type": record.error_type,
                "error_signature": record.error_signature,
                "error_category": record.error_category,
                "context_tags": record.context_tags,
                "initial_hypothesis": record.initial_hypothesis,
                "failed_strategies": record.failed_strategies,
                "learned_constraints": record.learned_constraints,
                "successful_strategy": None,
                "root_cause": None, "lesson": None,
                "outcome": "in_progress",
                "trials_count": len(record.trials),
            })

        bug_index_summary = []
        for bug in all_bugs:
            summary_line = (
                f"{bug['bug_id']} [{bug['status']}]: "
                f"{bug['error_category'] or 'Unknown'} - "
                f"{bug['error_signature'] or bug['error_type']} "
                f"(trials={bug['trials_count']})"
            )
            bug_index_summary.append(summary_line)

        prompt = {
            "Your Role": "RAG - Retrieve relevant bugs from historical memory",
            "Current Bug": {
                "Error Type": current_error_type,
                "Error Message": current_error_msg,
                "Code": current_code,
                "Plan Context": original_plan,
            },
            "Bug Index (Table of Contents)": bug_index_summary,
            "Detailed Bug Records": all_bugs,
            "Task": (
                "Select ALL relevant bug IDs that match the current error.\n"
                "Include ACTIVE bugs - their failed_strategies show what to AVOID!"
            ),
        }

        try:
            result = self._query(
                system_message=prompt,
                user_message=None,
                func_spec=retrieve_spec,
                model=self.model,
                temperature=self.temperature,
            )
            bug_ids = result.get("selected_bug_ids", [])
            selected = []
            for b in bug_ids:
                if b in self.bug_records:
                    selected.append(self.bug_records[b])
                elif b in self.active_bugs:
                    selected.append(self.active_bugs[b])
            return {
                "selected_bugs": selected,
                "reasoning": result.get("reasoning", ""),
                "key_patterns": result.get("key_patterns", []),
            }
        except Exception as e:
            logger.error("Bug retrieval failed: %s", e)
            matched = [
                r
                for r in list(self.bug_records.values()) + list(self.active_bugs.values())
                if (r.error_type or "").lower() == (current_error_type or "").lower()
            ]
            return {
                "selected_bugs": matched[:3],
                "reasoning": f"Fallback: matched by error type ({len(matched)} found)",
                "key_patterns": [],
            }

    def format_context_for_actor(self, retrieval_result: dict) -> str:
        early_stopping = (
            "PROVEN FIX:\n"
            "- LightGBM early stopping: callbacks=[lgb.early_stopping(stopping_rounds=N)]\n"
            "- XGBoost early stopping: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)"
        )

        if not retrieval_result.get("selected_bugs"):
            return early_stopping

        lines = []
        banned_items = []
        for r in retrieval_result["selected_bugs"]:
            if r.failed_strategies:
                for strategy in r.failed_strategies:
                    banned_items.append(strategy)
            elif r.error_signature:
                banned_items.append(r.error_signature)

        banned_items.append("'early_stopping_rounds' in LGBMClassifier.fit() / LGBMRegressor.fit() / lgb.train()")
        banned_items.append("'early_stopping_rounds' or 'callbacks' in XGBClassifier.fit() / XGBRegressor.fit()")

        proven_fixes = []
        for r in retrieval_result["selected_bugs"]:
            if r.final_outcome == "success" and r.successful_strategy:
                proven_fixes.append(r.successful_strategy)

        if banned_items:
            lines.append("BANNED (will crash):")
            for item in banned_items:
                lines.append(f"- {item}")

        proven_fixes.append("LightGBM early stopping: callbacks=[lgb.early_stopping(stopping_rounds=N)]")
        proven_fixes.append("XGBoost early stopping: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)")

        if proven_fixes:
            lines.append("")
            lines.append("PROVEN FIX:")
            for fix in proven_fixes:
                lines.append(f"- {fix}")

        result = "\n".join(lines)
        if self.advice_budget_chars > 0 and len(result) > self.advice_budget_chars:
            return result[: self.advice_budget_chars].rstrip()
        return result

    def get_guidance(self, plan: str = "", current_node: Optional["Node"] = None) -> str:
        if not current_node:
            return ""
        if not self.bug_records and not self.active_bugs:
            return "No historical bugs to learn from yet."

        try:
            full_error_msg = "".join(current_node._term_out) if hasattr(current_node, "_term_out") and current_node._term_out else (current_node.term_out or "")
            retrieval = self.retrieve_relevant_context(
                current_error_type=current_node.exc_type or "Unknown",
                current_error_msg=full_error_msg,
                current_code=current_node.code,
                original_plan=plan,
            )
            if not retrieval.get("selected_bugs"):
                return "No relevant historical bugs found."
            return self.format_context_for_actor(retrieval)
        except Exception as e:
            logger.error("get_guidance failed: %s", e)
            return ""

    def get_prevention_guidance(self, mode: str = "executive", journal: Optional["Journal"] = None) -> str:
        early_stopping_guidance = (
            "BANNED:\n"
            "- 'early_stopping_rounds' parameter in LGBMClassifier.fit() / LGBMRegressor.fit() / lgb.train() (causes TypeError)\n"
            "  USE: callbacks=[lgb.early_stopping(stopping_rounds=N)]\n"
            "- 'early_stopping_rounds' or 'callbacks' in XGBClassifier.fit() / XGBRegressor.fit() (causes TypeError)\n"
            "  USE: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)"
        )

        if not self.bug_records and not self.active_bugs:
            return early_stopping_guidance

        # Only re-distill when bug state has changed (dirty flag set by learn_from_bug)
        if not self._guidance_dirty and self._distilled_guidance:
            logger.debug("Using cached distilled guidance (no bug state change)")
            return self._distilled_guidance

        # Lock so only one thread distills at a time
        with self._guidance_lock:
            # Double-check after acquiring lock (another thread may have just distilled)
            if not self._guidance_dirty and self._distilled_guidance:
                return self._distilled_guidance

            raw_content = self._render_world_model(journal=journal)
            distilled = self._distill_guidance(raw_content)
            self._distilled_guidance = distilled
            self._guidance_dirty = False

            if self.save_dir:
                try:
                    path = self.save_dir / "distilled_guidance.md"
                    path.write_text(distilled)
                except Exception as e:
                    logger.error("Failed to save distilled guidance: %s", e)

            logger.info("=== DISTILLED GUIDANCE BEING PASSED ===")
            logger.info("%s", distilled[:2000] if len(distilled) > 2000 else distilled)
            logger.info("=== END DISTILLED GUIDANCE ===")

            return distilled

    def _distill_guidance(self, raw_world_model: str) -> str:
        if not raw_world_model.strip():
            return ""

        all_bugs = list(self.bug_records.values()) + list(self.active_bugs.values())
        if not all_bugs:
            return ""

        proven_failures = []
        proven_successes = []

        for record in all_bugs:
            if record.failed_strategies:
                for strategy in record.failed_strategies:
                    if strategy and len(strategy) > 10:
                        proven_failures.append({
                            "strategy": strategy,
                            "error_type": record.error_type,
                            "bug_id": record.bug_id
                        })
            elif record.error_signature:
                proven_failures.append({
                    "strategy": record.error_signature,
                    "error_type": record.error_type,
                    "bug_id": record.bug_id
                })

            if record.final_outcome == "success" and record.successful_strategy:
                proven_successes.append({
                    "strategy": record.successful_strategy,
                    "error_type": record.error_type,
                    "bug_id": record.bug_id
                })

        if not proven_failures:
            return ""

        prompt = {
            "Task": "Extract BANNED PATTERNS from these crash errors",
            "Instructions": [
                "For each failure, describe the banned pattern as text",
                "Include the error type (TypeError, ValueError, etc.)",
                "ONLY include fix_syntax if it appears in Proven Successes list",
            ],
            "Proven Failures (extract banned patterns)": [f["strategy"] for f in proven_failures],
            "Proven Successes (ONLY these can be fix_syntax)": [s["strategy"] for s in proven_successes] if proven_successes else [],
        }

        try:
            result = self._query(
                system_message=prompt,
                user_message=None,
                func_spec=distill_world_model_spec,
                model=self.model,
                temperature=0.0
            )

            patterns = result.get("patterns", [])
            if not patterns:
                return self._distill_guidance_fallback(proven_failures, proven_successes)

            lines = ["BANNED:"]
            for p in patterns:
                pattern = p.get("pattern", "")
                error_type = p.get("error_type", "")
                fix_syntax = p.get("fix_syntax", "")

                if pattern:
                    if error_type:
                        lines.append(f"- {pattern} (causes {error_type})")
                    else:
                        lines.append(f"- {pattern}")
                    if fix_syntax:
                        lines.append(f"  USE: {fix_syntax}")

            lines.append("- 'early_stopping_rounds' parameter in LGBMClassifier.fit() / LGBMRegressor.fit() / lgb.train() (causes TypeError)")
            lines.append("  USE: callbacks=[lgb.early_stopping(stopping_rounds=N)]")
            lines.append("- 'early_stopping_rounds' or 'callbacks' in XGBClassifier.fit() / XGBRegressor.fit() (causes TypeError)")
            lines.append("  USE: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)")

            return "\n".join(lines)

        except Exception as e:
            logger.warning("LLM distillation failed: %s. Using fallback.", e)
            return self._distill_guidance_fallback(proven_failures, proven_successes)

    def _distill_guidance_fallback(self, proven_failures: list, proven_successes: list) -> str:
        lines = ["BANNED:"]

        for failure in proven_failures:
            strategy = failure["strategy"]
            error_type = failure.get("error_type", "")
            if " -> " in strategy:
                strategy = strategy.split(" -> ")[0]
            strategy = strategy.replace("Call ", "").replace("Use ", "").replace("Try ", "").strip()
            strategy = re.sub(r'=\d+', '=...', strategy)
            strategy = re.sub(r'=True', '=...', strategy)
            strategy = re.sub(r'=False', '=...', strategy)
            if error_type:
                lines.append(f"- {strategy} (causes {error_type})")
            else:
                lines.append(f"- {strategy}")

        lines.append("- 'early_stopping_rounds' parameter in LGBMClassifier.fit() / LGBMRegressor.fit() / lgb.train() (causes TypeError)")
        lines.append("  USE: callbacks=[lgb.early_stopping(stopping_rounds=N)]")
        lines.append("- 'early_stopping_rounds' or 'callbacks' in XGBClassifier.fit() / XGBRegressor.fit() (causes TypeError)")
        lines.append("  USE: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)")

        return "\n".join(lines) if len(lines) > 1 else ""

    def get_statistics(self) -> dict:
        return {
            "total_bugs": len(self.bug_records),
            "active_bugs": len(self.active_bugs),
            "successful_fixes": sum(1 for r in self.bug_records.values() if r.final_outcome == "success"),
            "abandoned": sum(1 for r in self.bug_records.values() if r.final_outcome == "abandoned"),
        }

    # ═══════════════════════════════════════════════════════════
    # LLM-based summarization (writer)
    # ═══════════════════════════════════════════════════════════

    def _summarize_bug_start(self, record: BugRecord) -> dict:
        prompt = {
            "Role": "Extract error information from output",
            "Error Output": record.buggy_output,
            "Task": (
                "Extract ONLY what's in the error output:\n"
                "1. error_signature: Copy the EXACT error line\n"
                "2. error_category: TYPE_ERROR, VALUE_ERROR, FILE_NOT_FOUND, TIMEOUT, UNKNOWN, etc.\n"
                "3. context_tags: Library/function names mentioned in error\n"
                "4. is_unfixable: True only for timeout/OOM"
            ),
        }

        try:
            return self._query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_bug_start_spec,
                model=self.model,
                temperature=0.0,
            )
        except Exception as e:
            logger.error("Failed to summarize bug start: %s", e)
            sig = self._error_signature(record.buggy_output)
            return {
                "error_signature": sig or "Unknown: no traceback",
                "error_category": "UNKNOWN" if not sig else "OTHER",
                "context_tags": [],
                "is_unfixable": False,
            }

    def _summarize_trial_failure(self, trial: DebugTrial, record: BugRecord) -> dict:
        error_output = trial.error_output or ""
        error_sig = self._error_signature(error_output) or trial.error_type or "unknown error"

        prompt = {
            "Role": "Capture what was tried and what error occurred",
            "Debug Plan": trial.debug_plan,
            "Error Output": error_output[:1000],
            "Task": (
                "Extract:\n"
                "1. what_was_tried: Brief description of the approach\n"
                "2. actual_error: Copy the EXACT error message\n"
                "3. failed_strategy_summary: '<what tried> -> <error>'"
            ),
        }

        try:
            return self._query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_trial_failure_spec,
                model=self.model,
                temperature=0.0,
            )
        except Exception as e:
            logger.error("Failed to summarize trial failure: %s", e)
            plan_short = (trial.debug_plan or "").strip().split('\n')[0][:80]
            return {
                "what_was_tried": plan_short or "Unknown approach",
                "actual_error": error_sig,
                "failed_strategy_summary": f"{plan_short} -> {error_sig}",
            }

    def _summarize_trial_success(self, trial: DebugTrial, record: BugRecord) -> dict:
        prompt = {
            "Your Role": "Analyze why this debug attempt succeeded",
            "Original Bug": {
                "Error Type": record.error_type,
                "Error Signature": record.error_signature or "Unknown",
                "Original Error": record.buggy_output[:1000],
            },
            "Successful Debug Attempt": {
                "Attempt Number": trial.attempt_num,
                "Debug Plan": trial.debug_plan,
                "Code Executed": trial.code[:2000],
            },
            "Previous Failed Attempts": record.failed_strategies,
            "Task": "Extract why it worked, strategy summary, and key insight.",
        }

        try:
            result = self._query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_trial_success_spec,
                model=self.model,
                temperature=0.1,
            )
            why_worked = result.get("why_worked", "")
            if not why_worked or len(why_worked.strip()) < 20:
                raise ValueError("LLM returned empty or too-short why_worked")
            return result
        except Exception as e:
            logger.warning("Failed to summarize trial success for %s: %s", record.bug_id, e)
            why_worked = trial.why_worked or "Code executed successfully"
            plan_lines = trial.debug_plan.strip().split('\n') if trial.debug_plan else []
            strategy = plan_lines[0] if plan_lines else "Applied fix"
            return {
                "why_worked": why_worked,
                "successful_strategy_summary": strategy[:5000],
                "key_insight": f"Approach resolved {record.error_type}",
            }

    def _summarize_bug_record(self, record: BugRecord) -> dict:
        prompt = {
            "Your Role": "Extract structured lessons from a debugging session",
            "Original Bug": {
                "Error Type": record.error_type,
                "Buggy Code": record.buggy_code,
                "Error Output": record.buggy_output,
                "Original Plan": record.original_plan,
            },
            "All Debug Trials": [
                {
                    "attempt": t.attempt_num,
                    "plan": t.debug_plan,
                    "outcome": t.outcome,
                    "error": t.error_type,
                    "why_worked": t.why_worked,
                    "why_failed": t.why_failed,
                }
                for t in record.trials
            ],
            "Final Outcome": record.final_outcome,
            "Task": "Extract root cause, strategies, and lesson.",
        }

        try:
            return self._query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_bug_record_spec,
                model=self.model,
                temperature=self.temperature,
            )
        except Exception as e:
            logger.error("Failed to summarize bug record: %s", e)
            sig = self._error_signature(record.buggy_output)
            return {
                "root_cause": (
                    f"{record.error_type}: {sig}"
                    if sig
                    else f"{record.error_type} (summarization failed)"
                ),
                "successful_strategy": (
                    record.trials[-1].debug_plan.strip().splitlines()[0][:200]
                    if record.trials
                    and record.trials[-1].outcome == "success"
                    and record.trials[-1].debug_plan
                    else None
                ),
                "failed_strategies": [
                    (
                        t.debug_plan.strip().splitlines()[0][:200]
                        if t.debug_plan
                        else f"Attempt {t.attempt_num}"
                    )
                    for t in record.trials
                    if t.outcome == "failed"
                ][:5],
                "lesson": (
                    f"For `{record.error_type}` with `{sig}`, apply the successful fix pattern."
                    if sig
                    else f"For `{record.error_type}`, apply the successful fix pattern."
                ),
            }

    # ═══════════════════════════════════════════════════════════
    # Persistence + world model writing
    # ═══════════════════════════════════════════════════════════

    def _prune_bug_records(self) -> None:
        if len(self.bug_records) <= self.max_bug_records:
            return
        sorted_records = sorted(self.bug_records.values(), key=lambda r: r.timestamp)
        to_remove = sorted_records[: len(self.bug_records) - self.max_bug_records]
        for r in to_remove:
            self.bug_records.pop(r.bug_id, None)
            if self.delete_pruned_bug_files and self.save_dir:
                try:
                    (self.save_dir / f"{r.bug_id}.md").unlink(missing_ok=True)
                except Exception:
                    pass

    def _enforce_active_bug_limit(self) -> None:
        if self.max_active_bugs <= 0:
            return
        if len(self.active_bugs) < self.max_active_bugs:
            return
        oldest_bug_id, _ = sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp)[0]
        self._auto_abandon_bug_record(oldest_bug_id, reason=f"Exceeded max_active_bugs={self.max_active_bugs}")

    def _auto_abandon_bug_record(self, bug_id: str, reason: str) -> None:
        if bug_id not in self.active_bugs:
            return
        record = self.active_bugs[bug_id]
        record.final_outcome = "abandoned"
        if not record.root_cause:
            record.root_cause = f"Abandoned (memory safety valve): {reason}"
        if not record.lesson:
            record.lesson = (
                f"For `{record.error_type}`, repeated attempts did not resolve the issue; "
                f"abandon this path and try a different strategy/node. ({reason})"
            )
        self.bug_records[bug_id] = record
        self.active_bugs.pop(bug_id, None)
        self._prune_bug_records()
        self._save_bug_record(record, active=False)

    @staticmethod
    def _error_signature(output: str | None) -> str:
        if not output:
            return ""
        text = str(output).strip()
        if not text:
            return ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        candidates: list[str] = []
        for ln in lines:
            if "traceback" in ln.lower():
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b", ln) and ":" in ln:
                candidates.append(ln)
        if candidates:
            return candidates[-1][:300]

        for ln in reversed(lines[-50:]):
            if "state:" in ln.lower():
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b", ln) and ":" in ln:
                return ln[:300]

        for ln in reversed(lines[-50:]):
            if re.search(r"(Error|Exception)\b", ln) and "traceback" not in ln.lower():
                return ln[:300]

        return lines[0][:300] if lines else ""

    def format_active_trial_history(self, bug_id: str, *, max_trials: int = 5) -> str:
        record = self.active_bugs.get(bug_id)
        if record is None or not record.trials:
            return ""

        trials = record.trials[-max(1, int(max_trials)):]
        out: list[str] = []
        out.append(f"TRIAL HISTORY for `{bug_id}` -- NEVER repeat failed approaches:")
        for t in trials:
            plan_line = (t.debug_plan or "").strip().splitlines()[0:1]
            plan_short = plan_line[0].strip() if plan_line else "(no plan)"
            if t.outcome == "failed":
                why = (t.why_failed or t.error_output or t.error_type or "").strip()
                why = " ".join(why.split())[:240]
                out.append(f"- NEVER DO THIS (Attempt {t.attempt_num} CRASHED): {plan_short}")
                if why:
                    out.append(f"  - Will crash because: {why}")
            else:
                why = (t.why_worked or "").strip()
                why = " ".join(why.split())[:240]
                out.append(f"- USE THIS (Attempt {t.attempt_num} WORKED): {plan_short}")
                if why:
                    out.append(f"  - Why it works: {why}")
        return "\n".join(out).strip()

    def _render_world_model(self, journal: Optional["Journal"] = None) -> str:
        lines: list[str] = []
        lines.append(f"# World Model (Semantic Policy) -- v{self.world_model_version}")
        lines.append("")

        lines.append("## 1) Operator Priors (Observed)")
        if journal is None:
            lines.append("- (journal not provided)")
        else:
            nodes = journal.nodes
            total = len(nodes)
            buggy = sum(1 for n in nodes if n.is_buggy)
            ok = total - buggy

            def _count(stage: str, ok_only: bool) -> int:
                if ok_only:
                    return sum(1 for n in nodes if n.stage_name == stage and not n.is_buggy)
                return sum(1 for n in nodes if n.stage_name == stage)

            for stage in ("draft", "improve", "debug"):
                lines.append(
                    f"- `{stage}`: {_count(stage, False)} runs, {_count(stage, True)} valid"
                )
            lines.append(f"- Total: {total} runs, {ok} valid, {buggy} buggy")
        lines.append("")

        lines.append("## 2) What Worked (Evidence-Based)")
        worked = [
            r for r in self.bug_records.values() if r.final_outcome == "success" and r.successful_strategy
        ]
        if not worked:
            lines.append("- (no completed successful fixes yet)")
        else:
            for r in sorted(worked, key=lambda x: x.timestamp, reverse=True)[:15]:
                lines.append(
                    f"- IF `{r.error_type}` THEN `{r.successful_strategy}` (bug_id={r.bug_id})"
                )
        lines.append("")

        lines.append("## 3) What Failed (Evidence-Based)")
        failed_items: list[str] = []
        for r in sorted(self.bug_records.values(), key=lambda x: x.timestamp, reverse=True)[:30]:
            for s in r.failed_strategies or []:
                failed_items.append(f"- IF `{r.error_type}` THEN `{s}` fails (bug_id={r.bug_id})")
        if not failed_items:
            lines.append("- (no recorded failed strategies yet)")
        else:
            lines.extend(failed_items[:20])
        lines.append("")

        lines.append("## 4) Lessons (Compact)")
        lessons = [
            (r.error_type, r.lesson, r.bug_id)
            for r in sorted(self.bug_records.values(), key=lambda x: x.timestamp, reverse=True)
            if r.lesson
        ]
        if not lessons:
            lines.append("- (no lessons yet)")
        else:
            for et, lesson, bug_id in lessons[:20]:
                lines.append(f"- `{et}`: {lesson} ({bug_id})")
        lines.append("")

        if self.active_bugs:
            lines.append("## Active Bugs (In Progress)")
            for bug_id, r in sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp, reverse=True)[:10]:
                lines.append(
                    f"- {bug_id}: `{r.error_type}` (trials so far: {len(r.trials)})"
                )
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _save_bug_record(self, record: BugRecord, active: bool = False) -> None:
        if not self.save_dir:
            return

        try:
            status_tag = "ACTIVE" if active else ("COMPLETED" if record.final_outcome == "success" else "FAILED")
            filepath = self.save_dir / f"{record.bug_id}.md"

            content: list[str] = [
                f"# {record.bug_id} {status_tag}",
                "",
                "## Bug Information",
                f"**Error Type**: {record.error_type}",
                f"**Error Signature**: {record.error_signature or '(not yet summarized)'}",
                f"**Error Category**: {record.error_category or 'Unknown'}",
                f"**Outcome**: {record.final_outcome}",
                f"**Trials**: {len(record.trials)}",
                "",
            ]

            if record.initial_hypothesis:
                content += ["## Initial Hypothesis", record.initial_hypothesis, ""]
            if record.context_tags:
                content += ["## Context Tags", f"`{', '.join(record.context_tags)}`", ""]
            if record.successful_strategy:
                content += ["## Successful Strategy", record.successful_strategy, ""]
            if record.failed_strategies:
                content += ["## Failed Strategies", *[f"- {s}" for s in record.failed_strategies], ""]
            if record.root_cause:
                content += ["## Root Cause", record.root_cause, ""]
            if record.lesson:
                content += ["## Lesson", record.lesson, ""]

            content += ["## Debug Trials", ""]
            for trial in record.trials:
                content += [
                    f"### Attempt {trial.attempt_num}: {trial.outcome.upper()}",
                    "",
                    "**Plan**:",
                    "```",
                    trial.debug_plan or "No plan",
                    "```",
                    "",
                ]
                if trial.outcome == "failed":
                    content.append(f"**Why Failed**: {trial.why_failed or 'Not recorded'}")
                    if trial.error_type:
                        content.append(f"**Error Type**: {trial.error_type}")
                else:
                    content.append(f"**Why Worked**: {trial.why_worked or 'Not recorded'}")
                content += ["", "---", ""]

            filepath.write_text("\n".join(content))
        except Exception as e:
            logger.error("Failed to save bug record: %s", e)

    def _render_bug_index(self) -> str:
        """Render a BUG_INDEX.md with table of all bugs for quick RAG access."""
        lines: list = []
        lines.append("# Bug Index / Table of Contents")
        lines.append("")
        lines.append("Quick reference for all bugs encountered. Use this for RAG to find relevant bugs.")
        lines.append("")

        total = len(self.bug_records) + len(self.active_bugs)
        completed = len([r for r in self.bug_records.values() if r.final_outcome == "success"])
        abandoned = len([r for r in self.bug_records.values() if r.final_outcome == "abandoned"])
        active = len(self.active_bugs)
        lines.append(f"**Total Bugs**: {total} | **Completed**: {completed} | **Abandoned**: {abandoned} | **Active**: {active}")
        lines.append("")

        lines.append("| Bug ID | Status | Error Type | Error Signature | Category | Trials | Outcome |")
        lines.append("|--------|--------|------------|-----------------|----------|--------|---------|")

        for bug_id, record in sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp, reverse=True):
            sig = record.error_signature or self._error_signature(record.buggy_output) or "Unknown"
            sig = sig[:80] + "..." if len(sig) > 80 else sig
            category = record.error_category or "Unknown"
            lines.append(f"| {bug_id} | ACTIVE | {record.error_type} | {sig} | {category} | {len(record.trials)} | in_progress |")

        for bug_id, record in sorted(self.bug_records.items(), key=lambda t: t[1].timestamp, reverse=True):
            sig = record.error_signature or self._error_signature(record.buggy_output) or "Unknown"
            sig = sig[:80] + "..." if len(sig) > 80 else sig
            category = record.error_category or "Unknown"
            status = "OK" if record.final_outcome == "success" else "FAIL"
            lines.append(f"| {bug_id} | {status} | {record.error_type} | {sig} | {category} | {len(record.trials)} | {record.final_outcome} |")

        lines.append("")
        lines.append("## Quick Access by Category")
        lines.append("")
        by_category: dict = {}
        for bug_id, record in list(self.active_bugs.items()) + list(self.bug_records.items()):
            cat = record.error_category or "Unknown"
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(bug_id)
        for category in sorted(by_category.keys()):
            bugs = by_category[category]
            lines.append(f"- **{category}**: {', '.join(sorted(bugs))}")

        lines.append("")
        lines.append("## What Worked (Quick Reference)")
        lines.append("")
        for record in sorted(self.bug_records.values(), key=lambda r: r.timestamp, reverse=True):
            if record.final_outcome == "success" and record.successful_strategy:
                lines.append(f"- **{record.bug_id}** ({record.error_type}): {record.successful_strategy}")

        lines.append("")
        lines.append("## What Failed (Quick Reference)")
        lines.append("")
        for record in sorted(self.bug_records.values(), key=lambda r: r.timestamp, reverse=True)[:10]:
            if record.failed_strategies:
                lines.append(f"- **{record.bug_id}** ({record.error_type}):")
                for strategy in record.failed_strategies[:3]:
                    lines.append(f"  - {strategy}")

        return "\n".join(lines) + "\n"

    def _save_bug_index(self) -> None:
        """Save the bug index for quick RAG access."""
        if not self.save_dir:
            return
        try:
            content = self._render_bug_index()
            path = self.save_dir / "BUG_INDEX.md"
            path.write_text(content)
        except Exception as e:
            logger.error("Failed to save bug index: %s", e)

    def _save_world_model(self, journal: Optional["Journal"] = None) -> None:
        if not self.save_dir:
            return
        try:
            self.world_model_version += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            content = self._render_world_model(journal=journal)
            path = self.save_dir / "world_model_LATEST.md"
            path.write_text(f"<!-- updated: {ts} -->\n\n{content}")
            # Also update bug index
            self._save_bug_index()
        except Exception as e:
            logger.error("Failed to save world model: %s", e)
