"""Semantic error equivalence detection for error-aware backtracking.

When the same underlying error is seen repeatedly in a debug chain, continuing to
debug is wasteful. This module detects that situation and signals evaluation.py to
trigger an early backtrack so the search can explore a different branch instead.
"""

import logging
from typing import TYPE_CHECKING, cast

from llm import FunctionSpec, query

if TYPE_CHECKING:
    from engine.search_node import SearchNode

logger = logging.getLogger("MLEvolve")

error_comparison_func_spec = FunctionSpec(
    name="compare_errors",
    json_schema={
        "type": "object",
        "properties": {
            "is_same_error": {
                "type": "boolean",
                "description": (
                    "True if both errors represent the same underlying issue "
                    "(same root cause, same type of bug), even if the exact message differs. "
                    "False if they are fundamentally different errors."
                ),
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


def _get_term_out(node: "SearchNode") -> str | None:
    """Safely extract terminal output string from a node."""
    if node._term_out is None:
        return None
    if isinstance(node._term_out, list):
        return "".join(node._term_out)
    return str(node._term_out)


def are_errors_equivalent(error1: str, error2: str, agent) -> bool:
    """Determine if two terminal outputs represent the same underlying error.

    Uses an LLM to compare semantically (not string equality), so
    ``KeyError: 'feature_col'`` and ``KeyError: 'label'`` are treated as the
    same class of failure. Results are cached on ``agent._error_equiv_cache``
    to avoid repeated LLM calls for the same pair.

    Args:
        error1: Terminal output / error string from first node.
        error2: Terminal output / error string from second node.
        agent:  AgentSearch instance — provides LLM config and the cache dict.

    Returns:
        True if fixing one would likely fix the other.
    """
    if not error1 or not error2:
        return False

    if error1 == error2:
        return True

    # Use the tail of each output — that is where exceptions appear
    e1 = error1[-500:]
    e2 = error2[-500:]

    cache_key = frozenset([e1, e2])
    if cache_key in agent._error_equiv_cache:
        return agent._error_equiv_cache[cache_key]

    use_llm = getattr(agent.scfg, "use_error_equivalence_check", True)
    if not use_llm:
        # Fast fallback: check if both share the same first exception line
        result = e1.split("\n")[-1].strip() == e2.split("\n")[-1].strip()
        agent._error_equiv_cache[cache_key] = result
        return result

    prompt = (
        "Determine if these two error messages represent the same underlying bug.\n\n"
        f"Error 1:\n{e1}\n\n"
        f"Error 2:\n{e2}\n\n"
        "Instructions:\n"
        "- Consider the error TYPE (e.g. KeyError, ValueError, FileNotFoundError)\n"
        "- Consider the ROOT CAUSE (e.g. missing column, wrong data type, file path)\n"
        "- Ignore differences in line numbers, file paths, or variable values\n"
        "- Return true if fixing one would likely fix the other"
    )

    try:
        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=error_comparison_func_spec,
                model=agent.acfg.feedback.model,
                temperature=0.0,
                cfg=agent.cfg,
            ),
        )
        result = bool(response["is_same_error"])
        logger.info(
            f"[error_backtrack] equivalent={result}: {response['reasoning']}"
        )
    except Exception as exc:
        logger.warning(
            f"[error_backtrack] LLM comparison failed ({exc}), defaulting to False"
        )
        result = False

    agent._error_equiv_cache[cache_key] = result
    return result


def count_equivalent_errors_in_chain(node: "SearchNode", agent) -> int:
    """Count how many consecutive buggy ancestors share the same underlying error.

    Walks the parent chain from ``node`` upward (stopping at the virtual root or
    at the first ancestor whose error differs), counting ancestors whose error is
    semantically equivalent to ``node``'s error.

    Args:
        node:  Current (buggy) node being evaluated.
        agent: AgentSearch instance.

    Returns:
        Number of ancestors (not counting ``node`` itself) with an equivalent error.
    """
    if not node.is_buggy:
        return 0

    node_error = _get_term_out(node)
    if not node_error:
        return 0

    count = 0
    ancestor = node.parent

    while ancestor is not None and ancestor.stage != "root":
        if ancestor.is_buggy:
            ancestor_error = _get_term_out(ancestor)
            if ancestor_error and are_errors_equivalent(node_error, ancestor_error, agent):
                count += 1
            else:
                # Stop at first different error to avoid jumping over a genuine fix
                break
        ancestor = ancestor.parent

    return count


def should_error_backtrack(node: "SearchNode", agent) -> bool:
    """Return True if the error threshold is reached and we should backtrack.

    This fires when ``error_backtrack_threshold > 0`` AND the same semantic
    error has appeared at least ``threshold`` times in the ancestor chain,
    signalling that the current debug direction is stuck.

    Args:
        node:  Current (buggy) node.
        agent: AgentSearch instance (provides threshold and cache).

    Returns:
        True if early backtrack should be triggered.
    """
    threshold = getattr(agent.scfg, "error_backtrack_threshold", 0)
    if not threshold or threshold <= 0:
        return False

    if not node.is_buggy:
        return False

    count = count_equivalent_errors_in_chain(node, agent)
    if count >= threshold:
        logger.info(
            f"[error_backtrack] Node {node.id[:8]}: same error class seen {count + 1} time(s) "
            f"(threshold={threshold}) — triggering early backtrack"
        )
        return True
    return False
