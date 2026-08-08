"""
Bug Consultant v2: Memory Writer + Retrieval-Only Reader for Debugging History

Design goals (ICML submission aligned):
- In-context RL (no parameter updates): improve behavior via better context and action selection.
- World model writing: persist compact, policy-relevant "what works / what fails" rules (not raw logs).
- Separation of retrieval and execution: provide curated context to the code-generating actor.

This module is adapted from `aideml_submit/aide/bug_consultant_v2.py` with:
- stronger trial tracking (records failed trials instead of starting new bug records)
- deterministic world model writing (no extra LLM call required)

MLEvolve adapter changes (only):
- imports: llm instead of .backend; SearchNode instead of Node
- query() calls: added cfg=self.cfg parameter
- constructor: added cfg parameter
- thread safety: threading.Lock on write methods
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

from llm import FunctionSpec, query

if TYPE_CHECKING:
    from engine.search_node import SearchNode, Journal

logger = logging.getLogger("MLEvolve")


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
    error_signature: Optional[str] = None  # Specific error message (LLM-extracted)
    error_category: Optional[str] = None  # Category for grouping (API_MISUSE, etc.)
    initial_hypothesis: Optional[str] = None  # Preliminary root cause
    context_tags: list[str] = field(default_factory=list)  # Tags for semantic matching

    # Extracted / summarized fields (from trials and completion)
    root_cause: Optional[str] = None  # Final confirmed root cause
    successful_strategy: Optional[str] = None  # What worked
    failed_strategies: list[str] = field(default_factory=list)  # What didn't work (incremental)
    learned_constraints: list[str] = field(default_factory=list)  # Reusable rules from failures
    lesson: Optional[str] = None  # Actionable takeaway

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
                "description": "The ACTUAL root cause explanation - what technically caused this bug. Be specific and concise (1-2 sentences). Example: 'LGBMRegressor.fit() doesn't accept early_stopping_rounds parameter in sklearn API v3.0+, requires callbacks parameter instead'. NOT just 'TypeError' or repeating the error message.",
            },
            "successful_strategy": {
                "type": "string",
                "description": "Specific strategy that worked, in one concise sentence. Example: 'Use callbacks=[lgb.early_stopping(50)] instead of early_stopping_rounds parameter'. Focus on the actionable fix, not verbose explanation.",
            },
            "failed_strategies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific strategies that failed with WHY each failed. One sentence per strategy. Example: 'Tried early_stopping_rounds parameter → incompatible with sklearn API'. Keep each entry concise but informative.",
            },
            "lesson": {
                "type": "string",
                "description": "Actionable, reusable pattern that can prevent this exact mistake in future. One concise sentence. Example: 'For LightGBM sklearn API TypeError with early_stopping_rounds: use callbacks=[lgb.early_stopping(rounds)] instead'. Must be specific enough that seeing this lesson prevents repeating the mistake.",
            },
        },
        "required": ["root_cause", "lesson"],
    },
    description="Extract root cause, strategies, and reusable lessons from a completed bug record. Be SPECIFIC and CONCISE - the goal is to create actionable knowledge that prevents repeating the same mistakes.",
)

summarize_bug_start_spec = FunctionSpec(
    name="summarize_bug_start",
    json_schema={
        "type": "object",
        "properties": {
            "error_signature": {
                "type": "string",
                "description": "EXACT error message from output. Copy the actual error line like 'TypeError: fit() got unexpected keyword argument'. If no error in output, use 'Unknown: no traceback'.",
            },
            "error_category": {
                "type": "string",
                "description": "Category: TYPE_ERROR, VALUE_ERROR, ATTRIBUTE_ERROR, KEY_ERROR, INDEX_ERROR, IMPORT_ERROR, FILE_NOT_FOUND, TIMEOUT, UNKNOWN, or OTHER",
            },
            "context_tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Library/function names from the error. Example: ['lightgbm', 'fit', 'early_stopping']. Only include what's actually mentioned in the error.",
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
                "description": "Brief description of what approach was attempted. Example: 'Used callbacks parameter in fit()'",
            },
            "actual_error": {
                "type": "string",
                "description": "The EXACT error message from output. Copy the error line.",
            },
            "failed_strategy_summary": {
                "type": "string",
                "description": "Format: '<what was tried> → <actual error>'. Example: 'Used callbacks in fit() → TypeError: unexpected keyword argument callbacks'",
            },
        },
        "required": ["what_was_tried", "actual_error", "failed_strategy_summary"],
    },
    description="Summarize what was tried and what error occurred. NO suggestions for fixes - only capture the failure.",
)

summarize_trial_success_spec = FunctionSpec(
    name="summarize_trial_success",
    json_schema={
        "type": "object",
        "properties": {
            "why_worked": {
                "type": "string",
                "description": "Specific reason WHY this approach succeeded. Be concise (1-2 sentences). Example: 'Using callbacks=[lgb.early_stopping(50)] with eval_set parameter correctly implements early stopping in LightGBM sklearn API'.",
            },
            "successful_strategy_summary": {
                "type": "string",
                "description": "One-line summary of successful strategy - what fixed the bug. Example: 'Replace early_stopping_rounds with callbacks=[lgb.early_stopping(rounds)] + eval_set'. Keep concise and actionable.",
            },
            "key_insight": {
                "type": "string",
                "description": "Key insight that made this work - the core understanding. One concise sentence. Example: 'LightGBM sklearn API requires callback-based early stopping with explicit validation set'.",
            },
        },
        "required": ["why_worked", "successful_strategy_summary", "key_insight"],
    },
    description="Summarize a successful debug trial for RL. Extract why it worked, strategy summary, and key insight. Be concise but complete.",
)

extract_error_signature_spec = FunctionSpec(
    name="extract_error_signature",
    json_schema={
        "type": "object",
        "properties": {
            "error_signature": {
                "type": "string",
                "description": "Compact, specific error signature. Example: 'TypeError: fit() got an unexpected keyword argument early_stopping_rounds'. One line, be precise.",
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

format_context_spec = FunctionSpec(
    name="format_debug_context",
    json_schema={
        "type": "object",
        "properties": {
            "formatted_context": {
                "type": "string",
                "description": (
                    "Organized context in clean markdown format with sections: "
                    "Overview, What Worked, What Failed, Key Patterns, Lessons."
                ),
            },
            "summary": {
                "type": "string",
                "description": "One paragraph summary of key insights.",
            },
        },
        "required": ["formatted_context", "summary"],
    },
    description="Organize selected historical bugs into a clear markdown context for the actor.",
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
                            "description": "State what the code must NOT do — use imperative action language, not description of the symptom. Good: \"using 'sparse' parameter in OneHotEncoder\", \"passing 'verbose' to LGBMClassifier.fit()\", \"accessing column 'sat_cnr_l1_mean' which does not exist in dataset\". Bad: \"Missing feature 'sat_cnr_l1_mean'\", \"sparse parameter issue\".",
                        },
                        "error_type": {
                            "type": "string",
                            "description": "The error type. Example: 'TypeError', 'ValueError'",
                        },
                        "fix_syntax": {
                            "type": "string",
                            "description": "Proven fix as SHORT code syntax (ONLY from Proven Successes list). Example: 'sparse_output=False', 'callbacks=[lgb.log_evaluation()]'. Leave empty if no proven fix exists.",
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

learn_conditional_rule_spec = FunctionSpec(
    name="generate_conditional_rule",
    json_schema={
        "type": "object",
        "properties": {
            "conditional_rule": {
                "type": "string",
                "description": "The conditional BANNED rule. Format: 'When the existing code has [PATTERN]: BANNED: [TRIGGER] (causes [ERROR_TYPE])'"
            }
        },
        "required": ["conditional_rule"]
    },
    description="Generate a conditional BANNED rule for a parent code pattern"
)

# ═══════════════════════════════════════════════════════════
# Bug Consultant Class
# ═══════════════════════════════════════════════════════════


class BugConsultant:
    """
    Memory writer + retrieval-only reader for debugging history.

    Responsibilities:
    - Write: store bug records, trials, and a compact world model (markdown).
    - Read: retrieve and format relevant historical bugs as curated context.

    Not responsible for:
    - Writing code (that's the actor's job).
    - Deciding what to do (search policy remains in the agent).
    """

    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.3,
        save_dir: Optional[Path] = None,
        max_bug_records: int = 50,
        advice_budget_chars: int = 200000,
        max_active_bugs: int = 200,
        max_trials_per_bug: int = 20,
        delete_pruned_bug_files: bool = False,
        cfg=None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_bug_records = max_bug_records
        self.advice_budget_chars = int(advice_budget_chars)
        self.max_active_bugs = max_active_bugs
        self.max_trials_per_bug = max_trials_per_bug
        self.delete_pruned_bug_files = delete_pruned_bug_files
        self.cfg = cfg
        self._write_lock = threading.Lock()
        self.parent_conditional_rules: dict = {}  # {parent_id: str}
        self._global_conditional_rules: list = []  # all learned conditional rules (global)

        self.save_dir = save_dir
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.bug_records: dict[str, BugRecord] = {}
        self.active_bugs: dict[str, BugRecord] = {}

        # World model version counter (increments on write)
        self.world_model_version: int = 0

        # Distilled guidance cache (invalidated when content actually changes)
        self._distilled_guidance: str = ""
        self._distilled_content_hash: str = ""  # Hash of input content when last distilled
        self._distilled_version: int = -1  # Version when last distilled (backup)

        logger.info("🗂️  BugConsultant v2 initialized (model: %s)", model)

    # ═══════════════════════════════════════════════════════════
    # Writer: Bug record lifecycle
    # ═══════════════════════════════════════════════════════════

    def start_bug_record(self, node: "SearchNode") -> str:
        if node.step is None:
            raise ValueError("Cannot start bug record: node.step is None")

        self._enforce_active_bug_limit()

        bug_id = f"bug_{node.step}"
        # Use full output (not truncated) for bug analysis
        full_output = "".join(node._term_out) if hasattr(node, "_term_out") else node.term_out
        record = BugRecord(
            bug_id=bug_id,
            original_node_step=node.step,
            error_type=node.exc_type or "Unknown",
            buggy_code=node.code,
            buggy_output=full_output,
            original_plan=node.plan or "",
        )

        # Stage 1: LLM summarization at bug start (for RAG)
        # This also determines if the error is unfixable (timeout/OOM/killed)
        logger.debug("Summarizing bug start for %s", bug_id)
        summary = self._summarize_bug_start(record)
        record.error_signature = summary.get("error_signature")
        record.error_category = summary.get("error_category")
        record.initial_hypothesis = summary.get("initial_hypothesis")
        record.context_tags = summary.get("context_tags", [])

        # Check if LLM determined this is unfixable (timeout/OOM/killed)
        if summary.get("is_unfixable", False):
            record.is_dead = True
            record.final_outcome = "dead"
            record.lesson = f"Unfixable error: {record.initial_hypothesis}"
            self.bug_records[bug_id] = record
            logger.info("💀 Created DEAD bug record: %s (unfixable: %s)", bug_id, record.error_category)
            self._save_bug_record(record, active=False)
            self._save_world_model()
            return ""  # Return empty - no active bug to debug

        self.active_bugs[bug_id] = record
        logger.info("📝 Started bug record: %s (category: %s)", bug_id, record.error_category)
        self._save_bug_record(record, active=True)
        return bug_id

    def record_trial(
        self,
        bug_id: str,
        node: "SearchNode",
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
        # Use full output (not truncated) for trial error analysis
        full_output = None
        if outcome == "failed":
            full_output = "".join(node._term_out) if hasattr(node, "_term_out") else node.term_out
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

        # Stage 2: LLM summarization for incremental RL learning
        # This is THE KEY to preventing retrying failed strategies and reusing successful ones
        if outcome == "failed":
            logger.debug("Summarizing failed trial #%s for %s", trial.attempt_num, bug_id)
            summary = self._summarize_trial_failure(trial, record)

            # Store LLM-extracted insights in the trial
            trial.why_failed = summary.get("actual_error", trial.why_failed)

            # Add to failed strategies (for RL: prevent retrying)
            failed_strategy = summary.get("failed_strategy_summary", "")
            if failed_strategy and failed_strategy not in record.failed_strategies:
                record.failed_strategies.append(failed_strategy)

            logger.info("📊 Recorded FAILED trial #%s for %s: %s", trial.attempt_num, bug_id, failed_strategy)

        elif outcome == "success":
            logger.debug("Summarizing successful trial #%s for %s", trial.attempt_num, bug_id)
            summary = self._summarize_trial_success(trial, record)

            # Store LLM-extracted insights in the trial
            trial.why_worked = summary.get("why_worked", trial.why_worked)

            # Store successful strategy (for RL: reuse this approach)
            successful_strategy = summary.get("successful_strategy_summary", "")
            if successful_strategy:
                record.successful_strategy = successful_strategy

            logger.info("📊 Recorded SUCCESS trial #%s for %s: %s", trial.attempt_num, bug_id, successful_strategy)

        self._save_bug_record(record, active=True)

        # Safety valve: cap attempts per bug (rare; defaults are intentionally high).
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

        # Extract lessons with one LLM call (fallback to minimal content).
        summary = self._summarize_bug_record(record)
        record.root_cause = summary.get("root_cause") or record.root_cause
        record.successful_strategy = summary.get("successful_strategy") or record.successful_strategy
        record.failed_strategies = summary.get("failed_strategies") or record.failed_strategies
        record.lesson = summary.get("lesson") or record.lesson

        self.bug_records[bug_id] = record
        self.active_bugs.pop(bug_id, None)
        self._prune_bug_records()

        logger.info("✅ Completed bug record %s", bug_id)
        self._save_bug_record(record, active=False)
        self._save_world_model(journal=journal)

    def abandon_bug_record(self, bug_id: str, *, journal: Optional["Journal"] = None) -> None:
        self.complete_bug_record(bug_id, outcome="abandoned", journal=journal)

    # ═══════════════════════════════════════════════════════════
    # Writer: Backward-compatible ingestion API
    # ═══════════════════════════════════════════════════════════

    def learn_from_bug(self, node: "SearchNode", journal: Optional["Journal"] = None) -> None:
        """
        Update bug memory from a node outcome.

        Key behavior (fixes a weakness in the reference implementation):
        - If node has a buggy parent, treat it as a TRIAL for the parent bug (success or failure).
        - Only start a new bug record when the node is buggy AND its parent is not buggy.

        Thread-safe: serialized via _write_lock (MLEvolve runs parallel workers).
        """
        with self._write_lock:
            self._learn_from_bug_impl(node, journal)

    def _learn_from_bug_impl(self, node: "SearchNode", journal: Optional["Journal"] = None) -> None:
        if node.step is None:
            logger.warning("Skipping BugConsultant update: node.step is None")
            return

        # Trial path: any child of a buggy parent is a debug attempt on that parent bug.
        if node.parent is not None and getattr(node.parent, "is_buggy", False):
            parent_step = getattr(node.parent, "step", None)
            if parent_step is None:
                return
            parent_bug_id = f"bug_{parent_step}"
            if parent_bug_id not in self.active_bugs:
                # Ensure parent bug is tracked.
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
                self._save_world_model(journal=journal)
                return

            self.record_trial(
                parent_bug_id,
                node,
                outcome="success",
                why_worked=node.analysis or "Executed successfully",
            )
            self.complete_bug_record(parent_bug_id, outcome="success", journal=journal)
            return

        # New bug: buggy node without a buggy parent.
        if node.is_buggy:
            self.start_bug_record(node)
            self._save_world_model(journal=journal)
            return

        # Non-buggy node without buggy parent: nothing to learn.
        self._save_world_model(journal=journal)

    def ingest_journal(self, journal: "Journal") -> None:
        """Rebuild memory state from an existing journal (resume mode)."""
        for node in journal.nodes:
            try:
                self.learn_from_bug(node, journal=journal)
            except Exception:
                continue

    def learn_conditional_rule(
        self,
        parent_id: str,
        parent_code: str,
        child_diff: str,
        error_type: str,
        error_msg: str,
    ) -> None:
        """
        Learn a parent-specific conditional rule when a child crashes on a valid parent.

        Only called when parent was valid (no exc_type). Identifies latent vulnerabilities
        in parent code that the child's diff exposed.

        Thread-safe: serialized via _write_lock.
        """
        try:
            system_message = (
                "You are a code bug analyst. A child improvement crashed even though its parent ran fine. "
                "The parent has a latent vulnerability that the child's diff exposed. "
                "Identify what specific pattern in the PARENT code makes it vulnerable. "
                "Generate a conditional BANNED rule: "
                "'When the existing code has [SPECIFIC PATTERN]: BANNED: [WHAT TRIGGERS THE CRASH] (causes [ERROR_TYPE])'. "
                "Do NOT add a FIX or USE line — only what is proven to crash."
            )
            user_message = (
                f"## Parent Code (was valid, ran without error)\n```python\n{parent_code[:3000]}\n```\n\n"
                f"## Child Diff (what was added)\n```diff\n{child_diff}\n```\n\n"
                f"## Error\nType: {error_type}\nMessage: {error_msg[:500]}"
            )

            result = query(
                system_message=system_message,
                user_message=user_message,
                func_spec=learn_conditional_rule_spec,
                model=self.model,
                temperature=0.0,
                cfg=self.cfg,
            )
            rule = result.get("conditional_rule", "").strip()
            if rule:
                with self._write_lock:
                    self.parent_conditional_rules[parent_id] = rule
                    if rule not in self._global_conditional_rules:
                        self._global_conditional_rules.append(rule)
                logger.info(
                    "[BugConsultant] Learned conditional rule for parent %s: %s",
                    parent_id, rule[:120]
                )
        except Exception as e:
            logger.warning(
                "[BugConsultant] learn_conditional_rule failed for parent %s: %s",
                parent_id, e
            )

    def get_conditional_guidance(self, parent_id: str) -> str:
        """Return the parent-specific conditional rule, or empty string if none."""
        return self.parent_conditional_rules.get(parent_id, "")

    # ═══════════════════════════════════════════════════════════
    # Reader: Retrieval and formatting (curated context only)
    # ═══════════════════════════════════════════════════════════

    def retrieve_relevant_context(
        self,
        current_error_type: str,
        current_error_msg: str,
        current_code: str,
        original_plan: str,
    ) -> dict:
        # Include BOTH completed AND active bugs for RAG
        # Active bugs show what was tried but didn't work yet - crucial for RL!
        if not self.bug_records and not self.active_bugs:
            return {"selected_bugs": [], "reasoning": "No historical bugs yet", "key_patterns": []}

        all_bugs = []

        # Add completed bugs
        for bug_id, record in self.bug_records.items():
            all_bugs.append(
                {
                    "bug_id": bug_id,
                    "status": "completed",  # Mark as completed
                    "error_type": record.error_type,
                    # RAG fields for semantic matching
                    "error_signature": record.error_signature,  # Specific error message
                    "error_category": record.error_category,  # API_MISUSE, TYPE_ERROR, etc.
                    "context_tags": record.context_tags,  # Semantic tags
                    "initial_hypothesis": record.initial_hypothesis,  # Preliminary root cause
                    # RL fields
                    "successful_strategy": record.successful_strategy,
                    "failed_strategies": record.failed_strategies,
                    "learned_constraints": record.learned_constraints,  # Reusable rules
                    # Completion fields
                    "root_cause": record.root_cause,
                    "lesson": record.lesson,
                    "outcome": record.final_outcome,
                    "trials_count": len(record.trials),
                }
            )

        # Add ACTIVE bugs (what's being tried right now but not working)
        for bug_id, record in self.active_bugs.items():
            all_bugs.append(
                {
                    "bug_id": bug_id,
                    "status": "active",  # Mark as active
                    "error_type": record.error_type,
                    # RAG fields for semantic matching
                    "error_signature": record.error_signature,
                    "error_category": record.error_category,
                    "context_tags": record.context_tags,
                    "initial_hypothesis": record.initial_hypothesis,
                    # RL fields - what didn't work so far
                    "failed_strategies": record.failed_strategies,  # Critical: what was tried and failed
                    "learned_constraints": record.learned_constraints,  # Rules learned from failures
                    "successful_strategy": None,  # Not solved yet
                    "root_cause": None,  # Not confirmed yet
                    "lesson": None,  # Not complete yet
                    "outcome": "in_progress",
                    "trials_count": len(record.trials),
                }
            )

        # Create a table of contents view for the LLM to scan
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
            "How to Use This": (
                "1. SCAN the Bug Index to see all available bugs at a glance\n"
                "2. MATCH based on error signature, category, and tags\n"
                "3. SELECT the most relevant bugs\n\n"
                "Bug statuses:\n"
                "- COMPLETED: Has final solution and lessons\n"
                "- ACTIVE: Currently being debugged - shows what DIDN'T WORK (crucial for RL!)"
            ),
            "Task": (
                "[Quick extraction - respond directly]\n"
                "Select ALL relevant bug IDs that match the current error.\n"
                "Include bugs that:\n"
                "- Have the same or similar error signature\n"
                "- Have the same error category\n"
                "- Share context tags with the current error\n"
                "- Have failed_strategies that could apply to this bug\n\n"
                "Do NOT limit the count artificially - if 10 bugs are relevant, select all 10.\n"
                "CRITICAL: Include ACTIVE bugs - their failed_strategies show what to AVOID!"
            ),
        }

        try:
            result = query(
                system_message=prompt,
                user_message=None,
                func_spec=retrieve_spec,
                model=self.model,
                temperature=self.temperature,
                cfg=self.cfg,
            )
            bug_ids = result.get("selected_bug_ids", [])
            # Retrieve from BOTH completed AND active bugs
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
            # Fallback: match by error type from BOTH completed and active bugs
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
        """Format historical context using ONLY evidence-based information.

        V4: Plain format (86% success vs 31% scary format in ablation tests)
        - Simple "BANNED:" prefix for failed approaches
        - "PROVEN FIX:" only for bugs that were successfully fixed
        """
        # V5: Early stopping guidance (always include)
        early_stopping = (
            "PROVEN FIX:\n"
            "- LightGBM early stopping: callbacks=[lgb.early_stopping(stopping_rounds=N)]\n"
            "- XGBoost early stopping: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)"
        )

        if not retrieval_result.get("selected_bugs"):
            return early_stopping

        lines = []

        # V4: Collect all failed strategies AND original error signatures
        banned_items = []
        for r in retrieval_result["selected_bugs"]:
            if r.failed_strategies:
                for strategy in r.failed_strategies:
                    banned_items.append(strategy)
            elif r.error_signature:  # Bug crashed but no debug trial yet
                banned_items.append(r.error_signature)

        # V5: Always include banned early stopping patterns (debug context needs this too)
        banned_items.append("'early_stopping_rounds' in LGBMClassifier.fit() / LGBMRegressor.fit() / lgb.train()")
        banned_items.append("'early_stopping_rounds' or 'callbacks' in XGBClassifier.fit() / XGBRegressor.fit()")

        # V4: Collect proven fixes (only from successfully debugged bugs)
        proven_fixes = []
        for r in retrieval_result["selected_bugs"]:
            if r.final_outcome == "success" and r.successful_strategy:
                proven_fixes.append(r.successful_strategy)

        # V4: Plain format - BANNED first
        if banned_items:
            lines.append("BANNED (will crash):")
            for item in banned_items:
                lines.append(f"- {item}")

        # V5: Always include early stopping as proven fix
        proven_fixes.append("LightGBM early stopping: callbacks=[lgb.early_stopping(stopping_rounds=N)]")
        proven_fixes.append("XGBoost early stopping: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)")

        # V4: Only show PROVEN FIX if bug was successfully fixed
        if proven_fixes:
            lines.append("")
            lines.append("PROVEN FIX:")
            for fix in proven_fixes:
                lines.append(f"- {fix}")

        result = "\n".join(lines)

        # Safety valve: truncation for context length management
        if self.advice_budget_chars > 0 and len(result) > self.advice_budget_chars:
            return result[: self.advice_budget_chars].rstrip()

        return result

    def get_guidance(self, plan: str = "", current_node: Optional["SearchNode"] = None) -> str:
        """
        Convenience wrapper: retrieve relevant bugs and format them.

        Note: Only one LLM call (retrieve). Format is now deterministic to prevent embellishment.
        """
        if not current_node:
            return ""
        # Check BOTH completed and active bugs - active bugs show what didn't work!
        if not self.bug_records and not self.active_bugs:
            return "No historical bugs to learn from yet."

        try:
            # Use full output (not truncated) for better bug matching
            full_error_msg = "".join(current_node._term_out) if hasattr(current_node, "_term_out") else (current_node.term_out or "")
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
        """
        Returns distilled DO/DON'T guidance for draft/improve nodes.
        Uses LLM to convert verbose bug summaries into actionable code patterns.
        Caches result until bug content actually changes (content-hash based).
        """
        # V5: Always include early stopping guidance even when no bugs yet
        early_stopping_guidance = (
            "BANNED:\n"
            "- 'early_stopping_rounds' parameter in LGBMClassifier.fit() / LGBMRegressor.fit() / lgb.train() (causes TypeError)\n"
            "  USE: callbacks=[lgb.early_stopping(stopping_rounds=N)]\n"
            "- 'early_stopping_rounds' or 'callbacks' in XGBClassifier.fit() / XGBRegressor.fit() (causes TypeError)\n"
            "  USE: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)"
        )

        if not self.bug_records and not self.active_bugs:
            return early_stopping_guidance

        # Get raw world model content
        raw_content = self._render_world_model(journal=journal)

        # Compute content hash to detect actual changes
        content_hash = hashlib.md5(raw_content.encode()).hexdigest()

        # Check if cache is valid (content hasn't changed)
        with self._write_lock:
            current_global_rules = list(self._global_conditional_rules)
        cache_key = (content_hash, len(current_global_rules))
        if self._distilled_content_hash == cache_key and self._distilled_guidance:
            logger.debug("Using cached distilled guidance (no content change, %d conditional rules)", len(current_global_rules))
            return self._distilled_guidance

        # Distill into DO/DON'T format
        distilled = self._distill_guidance(raw_content)

        # Cache the result with content hash + conditional rule count
        self._distilled_guidance = distilled
        self._distilled_content_hash = cache_key
        self._distilled_version = self.world_model_version

        # Save to disk
        if self.save_dir:
            try:
                path = self.save_dir / "distilled_guidance.md"
                path.write_text(distilled)
                logger.info("Saved distilled guidance to %s", path)
            except Exception as e:
                logger.error("Failed to save distilled guidance: %s", e)

        # Append global conditional rules so every node sees them
        if current_global_rules:
            rules_text = "\n\n⚠️ CONDITIONAL BUG PATTERNS (apply to all future code):\n"
            for r in current_global_rules:
                rules_text += f"- {r}\n"
            distilled = distilled + rules_text
            self._distilled_guidance = distilled

        # Log what's being passed
        logger.info("=== DISTILLED GUIDANCE BEING PASSED ===")
        logger.info("%s", distilled[:2000] if len(distilled) > 2000 else distilled)
        logger.info("=== END DISTILLED GUIDANCE ===")

        return distilled

    def _distill_guidance(self, raw_world_model: str) -> str:
        """
        Create guidance from bug patterns using LLM-based distillation.

        Uses LLM to convert raw failed strategies into generalizable MUST NOT constraints.
        This is more robust than pattern matching because LLM understands context.

        CRITICAL PRINCIPLE: Only output what is PROVEN.
        - Failed strategies → "NEVER use parameter X" (proven to fail)
        - Successful strategies → "Use Y instead" (proven to work)
        - NO SUGGESTIONS for alternatives unless proven to work
        """
        if not raw_world_model.strip():
            return ""

        all_bugs = list(self.bug_records.values()) + list(self.active_bugs.values())
        if not all_bugs:
            return ""

        # Collect ALL proven failures and successes (no truncation)
        proven_failures = []
        proven_successes = []

        for record in all_bugs:
            # Include failed debug strategies
            if record.failed_strategies:
                for strategy in record.failed_strategies:
                    if strategy and len(strategy) > 10:
                        proven_failures.append({
                            "strategy": strategy,
                            "error_type": record.error_type,
                            "bug_id": record.bug_id
                        })
            # Also include original error signature for bugs with no debug trials yet
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

        # Deduplicate by strategy text before passing to LLM (same error_signature can appear
        # multiple times when the same bug recurs and is each time fixed on first try)
        seen_strategies = set()
        unique_failures = []
        for f in proven_failures:
            s = f["strategy"]
            if s not in seen_strategies:
                seen_strategies.add(s)
                unique_failures.append(f)
        proven_failures = unique_failures

        seen_success = set()
        unique_successes = []
        for s in proven_successes:
            k = s["strategy"]
            if k not in seen_success:
                seen_success.add(k)
                unique_successes.append(s)
        proven_successes = unique_successes

        # Use LLM to extract banned patterns
        prompt = {
            "Task": "Extract BANNED PATTERNS from these crash errors",
            "Instructions": [
                "For each failure, state what the code must NOT DO in imperative form — not what went wrong, but what action to avoid. Use action verbs: 'accessing X', 'using Y parameter', 'calling Z method'. BAD: 'Missing feature X'. GOOD: 'accessing column X which does not exist in dataset'.",
                "Include the error type (TypeError, ValueError, etc.)",
                "ONLY include fix_syntax if it appears in Proven Successes list - no speculation",
                "fix_syntax should be SHORT code syntax (e.g., 'sparse_output=False')",
                "IMPORTANT: Do NOT skip any error. Include dataset errors which help identify data-related assumptions that caused failures, and generalize the root cause into a reusable pattern.",
            ],
            "Proven Failures (extract banned patterns)": [f["strategy"] for f in proven_failures],
            "Proven Successes (ONLY these can be fix_syntax)": [s["strategy"] for s in proven_successes] if proven_successes else [],
        }

        try:
            result = query(
                system_message=prompt,
                user_message=None,
                func_spec=distill_world_model_spec,
                model=self.model,
                temperature=0.0,
                cfg=self.cfg,
            )

            # V5: Convert structured result to pattern-based format
            patterns = result.get("patterns", [])
            if not patterns:
                return self._distill_guidance_fallback(proven_failures, proven_successes)

            # V5: New format - "BANNED:" with pattern description and error type
            # Format: - 'sparse' parameter in OneHotEncoder (causes TypeError)
            #           USE: sparse_output=False
            lines = ["BANNED:"]

            for p in patterns:
                pattern = p.get("pattern", "")
                error_type = p.get("error_type", "")
                fix_syntax = p.get("fix_syntax", "")  # Only from proven successes

                if pattern:
                    # Format: pattern (causes ErrorType)
                    if error_type:
                        lines.append(f"- {pattern} (causes {error_type})")
                    else:
                        lines.append(f"- {pattern}")

                    # Only show USE: if proven fix exists
                    if fix_syntax:
                        lines.append(f"  USE: {fix_syntax}")

            # V5: Always include early stopping guidance (proven fix)
            lines.append("- 'early_stopping_rounds' parameter in LGBMClassifier.fit() / LGBMRegressor.fit() / lgb.train() (causes TypeError)")
            lines.append("  USE: callbacks=[lgb.early_stopping(stopping_rounds=N)]")
            lines.append("- 'early_stopping_rounds' or 'callbacks' in XGBClassifier.fit() / XGBRegressor.fit() (causes TypeError)")
            lines.append("  USE: xgboost.train(params, dtrain, evals=[(dval, 'val')], early_stopping_rounds=N)")

            return "\n".join(lines)

        except Exception as e:
            logger.warning("LLM distillation failed: %s. Using fallback.", e)
            return self._distill_guidance_fallback(proven_failures, proven_successes)

    def _distill_guidance_fallback(self, proven_failures: list, proven_successes: list) -> str:
        """Fallback: Simple deterministic format when LLM fails. V5: Pattern-based format."""
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

        # V5: Always include early stopping guidance (proven fix)
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
        """
        Stage 1: Extract error info from output for RAG retrieval.
        Simple extraction - NO speculation, NO hypotheses.
        """
        prompt = {
            "Role": "Extract error information from output",
            "Error Output": record.buggy_output,
            "Task": (
                "Extract ONLY what's in the error output:\n"
                "1. error_signature: Copy the EXACT error line (e.g., 'TypeError: fit() got unexpected...')\n"
                "2. error_category: TYPE_ERROR, VALUE_ERROR, FILE_NOT_FOUND, TIMEOUT, UNKNOWN, etc.\n"
                "3. context_tags: Library/function names mentioned in error\n"
                "4. is_unfixable: True only for timeout/OOM\n\n"
                "If no traceback in output, use error_signature='Unknown: no traceback' and category='UNKNOWN'"
            ),
        }

        try:
            return query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_bug_start_spec,
                model=self.model,
                temperature=0.0,
                cfg=self.cfg,
            )
        except Exception as e:
            logger.error("Failed to summarize bug start: %s", e)
            # Simple fallback
            sig = self._error_signature(record.buggy_output)
            return {
                "error_signature": sig or "Unknown: no traceback",
                "error_category": "UNKNOWN" if not sig else "OTHER",
                "context_tags": [],
                "is_unfixable": False,
            }

    def _summarize_trial_failure(self, trial: DebugTrial, record: BugRecord) -> dict:
        """
        Summarize a failed trial - capture what was tried and what error occurred.
        NO suggestions for fixes - only capture the failure.
        """
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
                "3. failed_strategy_summary: '<what tried> → <error>'\n\n"
                "NO suggestions for fixes - only capture what failed."
            ),
        }

        try:
            return query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_trial_failure_spec,
                model=self.model,
                temperature=0.0,
                cfg=self.cfg,
            )
        except Exception as e:
            logger.error("Failed to summarize trial failure: %s", e)
            # Simple fallback - just capture what happened
            plan_short = (trial.debug_plan or "").strip().split('\n')[0][:80]
            return {
                "what_was_tried": plan_short or "Unknown approach",
                "actual_error": error_sig,
                "failed_strategy_summary": f"{plan_short} → {error_sig}",
            }

    def _summarize_trial_success(self, trial: DebugTrial, record: BugRecord) -> dict:
        """
        Stage 2b: Summarize a successful debug trial for RL.
        Extracts why it worked, strategy summary, and key insight.
        """
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
                "Code Executed": trial.code[:2000],  # Show some code context
            },
            "Previous Failed Attempts": record.failed_strategies,
            "Task": (
                "[Quick extraction - respond directly]\n"
                "Analyze WHY this approach succeeded where others failed.\n\n"
                "Extract:\n"
                "1. WHY WORKED: Specific technical reason this succeeded (1-2 sentences)\n"
                "2. SUCCESSFUL STRATEGY SUMMARY: One-line fix summary\n"
                "3. KEY INSIGHT: Core understanding\n\n"
                "Focus on the MECHANISM of success, not just stating 'it worked'."
            ),
        }

        try:
            result = query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_trial_success_spec,
                model=self.model,
                temperature=0.1,
                cfg=self.cfg,
            )
            # Validate that we got meaningful content
            why_worked = result.get("why_worked", "")
            if not why_worked or len(why_worked.strip()) < 20:
                raise ValueError("LLM returned empty or too-short why_worked")
            return result
        except Exception as e:
            logger.warning("Failed to summarize trial success for %s: %s. Using best-effort fallback.", record.bug_id, e)
            # Better fallback: extract from available context
            why_worked = trial.why_worked or "Code executed successfully"

            # Try to extract meaningful info from the debug plan
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
            "Task": (
                "[Quick extraction - respond directly]\n"
                "Extract structured knowledge:\n\n"
                "1. ROOT CAUSE: What caused this bug?\n"
                "2. SUCCESSFUL STRATEGY: What worked? (if any)\n"
                "3. FAILED STRATEGIES: What didn't work?\n"
                "4. LESSON: One-line reusable pattern\n\n"
                "Be concise but complete. This becomes searchable knowledge."
            ),
        }

        try:
            return query(
                system_message=prompt,
                user_message=None,
                func_spec=summarize_bug_record_spec,
                model=self.model,
                temperature=self.temperature,
                cfg=self.cfg,
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
                    f"For `{record.error_type}` with `{sig}`, apply the successful fix pattern from bug `{record.bug_id}`."
                    if sig
                    else f"For `{record.error_type}`, apply the successful fix pattern from bug `{record.bug_id}`."
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
                    (self.save_dir / f"{r.bug_id}.md").unlink(missing_ok=True)  # type: ignore[arg-type]
                except Exception:
                    pass

    def _enforce_active_bug_limit(self) -> None:
        """Safety valve: avoid unbounded growth of active bugs (rare; defaults are high)."""
        if self.max_active_bugs <= 0:
            return
        if len(self.active_bugs) < self.max_active_bugs:
            return
        # Evict oldest active bug record without an extra LLM summarization call.
        oldest_bug_id, _ = sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp)[0]
        self._auto_abandon_bug_record(oldest_bug_id, reason=f"Exceeded max_active_bugs={self.max_active_bugs}")

    def _auto_abandon_bug_record(self, bug_id: str, reason: str) -> None:
        """
        Mark an active bug record as abandoned due to memory limits.
        This avoids extra LLM calls (we record a concise lesson directly).
        """
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

    def _extract_error_signature_llm(self, output: str) -> tuple[str, str]:
        """
        Use LLM to extract a meaningful error signature and category from execution output.
        Returns: (error_signature, error_category)
        """
        if not output or len(output.strip()) < 10:
            return ("No error output", "UNKNOWN")

        try:
            result = query(
                system_message={
                    "Your Role": "Extract error signature from execution output",
                    "Execution Output": output,
                    "Task": (
                        "[Quick extraction - respond directly]\n"
                        "Extract the SPECIFIC error message that would help identify this exact bug.\n"
                        "Examples:\n"
                        "- 'TypeError: fit() got an unexpected keyword argument early_stopping_rounds'\n"
                        "- 'ValueError: Input contains NaN, infinity or a value too large for dtype(float64)'\n"
                        "- 'KeyError: \"['feature_x'] not found in axis\"'\n\n"
                        "Do NOT return generic messages like 'Traceback (most recent call last):'"
                    ),
                },
                user_message=None,
                func_spec=extract_error_signature_spec,
                model=self.model,
                temperature=0.0,
                cfg=self.cfg,
            )
            return (result.get("error_signature", "Unknown error"), result.get("error_category", "UNKNOWN"))
        except Exception as e:
            logger.debug("LLM error signature extraction failed: %s, falling back to heuristic", e)
            return (self._error_signature(output), "UNKNOWN")

    def _save_bug_record(self, record: BugRecord, active: bool = False) -> None:
        if not self.save_dir:
            return

        try:
            status_tag = "🔴 ACTIVE" if active else ("✅ COMPLETED" if record.final_outcome == "success" else "❌ FAILED")
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

            # RAG fields (for semantic matching)
            if record.initial_hypothesis:
                content += ["## Initial Hypothesis", record.initial_hypothesis, ""]
            if record.context_tags:
                content += ["## Context Tags", f"`{', '.join(record.context_tags)}`", ""]

            # RL fields (what worked / what didn't)
            if record.successful_strategy:
                content += ["## Successful Strategy", record.successful_strategy, ""]
            if record.failed_strategies:
                content += ["## Failed Strategies", *[f"- {s}" for s in record.failed_strategies], ""]
            if record.learned_constraints:
                content += ["## Learned Constraints", *[f"- {c}" for c in record.learned_constraints], ""]

            # Completion fields
            if record.root_cause:
                content += ["## Root Cause", record.root_cause, ""]
            if record.lesson:
                content += ["## Lesson", record.lesson, ""]

            # Trial history
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

    @staticmethod
    def _error_signature(output: str | None) -> str:
        """
        Extract a compact, searchable error signature from a traceback/log.
        Prefers the final exception line when present.
        """
        if not output:
            return ""
        text = str(output).strip()
        if not text:
            return ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # Heuristic: prefer lines that look like an exception summary (even if the log is truncated).
        candidates: list[str] = []
        for ln in lines:
            if "traceback" in ln.lower():
                continue
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b", ln) and ":" in ln:
                candidates.append(ln)
        if candidates:
            return candidates[-1][:300]

        # Prefer the final "XError: message" line.
        for ln in reversed(lines[-50:]):
            if "state:" in ln.lower():
                continue
            # Match "TypeError: ...", "ValueError: ...", "Exception: ...", etc.
            # (Avoid word-boundary pitfalls like "TypeError" where "Error" isn't a separate word.)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*(Error|Exception)\b", ln) and ":" in ln:
                return ln[:300]

        # Secondary: find any exception-ish line even without a colon (rare).
        for ln in reversed(lines[-50:]):
            if re.search(r"(Error|Exception)\b", ln) and "traceback" not in ln.lower():
                return ln[:300]

        return lines[0][:300] if lines else ""

    def format_active_trial_history(self, bug_id: str, *, max_trials: int = 5) -> str:
        """
        Deterministic, no-LLM summary of active attempts on the current bug.
        Used to prevent repeating known-bad strategies during multi-attempt debugging.
        """
        record = self.active_bugs.get(bug_id)
        if record is None or not record.trials:
            return ""

        trials = record.trials[-max(1, int(max_trials)) :]
        out: list[str] = []
        out.append(f"TRIAL HISTORY for `{bug_id}` — NEVER repeat failed approaches:")
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
        lines.append(f"# World Model (Semantic Policy) — v{self.world_model_version}")
        lines.append("")

        # 1) Operator priors (derived from journal if provided)
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
                    f"- `{stage}`: { _count(stage, False) } runs, { _count(stage, True) } valid"
                )
            lines.append(f"- Total: {total} runs, {ok} valid, {buggy} buggy")
        lines.append("")

        # 2/3) What worked / failed from bug records
        lines.append("## 2) What Worked (Evidence-Based)")
        worked = [
            r for r in self.bug_records.values() if r.final_outcome == "success" and r.successful_strategy
        ]
        if not worked:
            lines.append("- (no completed successful fixes yet)")
        else:
            for r in sorted(worked, key=lambda x: x.timestamp, reverse=True)[:15]:
                lines.append(
                    f"- IF `{r.error_type}` THEN `{r.successful_strategy}` (bug_id={r.bug_id}, trials={len(r.trials)})"
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

    def _render_bug_index(self) -> str:
        """
        Render a table of contents / index of all bugs for RAG.
        This allows quick scanning and semantic matching of bugs.
        """
        lines: list[str] = []
        lines.append("# Bug Index / Table of Contents")
        lines.append("")
        lines.append("Quick reference for all bugs encountered. Use this for RAG to find relevant bugs.")
        lines.append("")

        # Summary stats
        total = len(self.bug_records) + len(self.active_bugs)
        completed = len([r for r in self.bug_records.values() if r.final_outcome == "success"])
        abandoned = len([r for r in self.bug_records.values() if r.final_outcome == "abandoned"])
        active = len(self.active_bugs)

        lines.append(f"**Total Bugs**: {total} | **Completed**: {completed} | **Abandoned**: {abandoned} | **Active**: {active}")
        lines.append("")

        # Table header
        lines.append("| Bug ID | Status | Error Type | Error Signature | Category | Trials | Outcome |")
        lines.append("|--------|--------|------------|-----------------|----------|--------|---------|")

        # Active bugs first
        for bug_id, record in sorted(self.active_bugs.items(), key=lambda t: t[1].timestamp, reverse=True):
            sig = record.error_signature or self._error_signature(record.buggy_output) or "Unknown"
            sig = sig[:80] + "..." if len(sig) > 80 else sig
            category = record.error_category or "Unknown"
            lines.append(
                f"| {bug_id} | 🔴 ACTIVE | {record.error_type} | {sig} | {category} | {len(record.trials)} | in_progress |"
            )

        # Completed bugs
        for bug_id, record in sorted(self.bug_records.items(), key=lambda t: t[1].timestamp, reverse=True):
            sig = record.error_signature or self._error_signature(record.buggy_output) or "Unknown"
            sig = sig[:80] + "..." if len(sig) > 80 else sig
            category = record.error_category or "Unknown"
            status = "✅" if record.final_outcome == "success" else "❌"
            lines.append(
                f"| {bug_id} | {status} | {record.error_type} | {sig} | {category} | {len(record.trials)} | {record.final_outcome} |"
            )

        lines.append("")
        lines.append("## Quick Access by Category")
        lines.append("")

        # Group by category
        by_category: dict[str, list[str]] = {}
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
