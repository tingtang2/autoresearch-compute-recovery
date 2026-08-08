"""HPO score: LLM-rated thoroughness of hyperparameter tuning in a successful node.

Returns int in {0,1,2,3}. Consumed by engine/evaluation.py reward shaping,
gated by scfg.use_hpo_score. Buggy nodes are filtered upstream so this is
only invoked on successfully-executed nodes.

Cached on agent._hpo_score_cache by node.id so the dual call site
(get_node_reward + get_normalized_reward) only triggers one LLM call.
"""

import logging
from typing import cast

from llm import FunctionSpec, query

logger = logging.getLogger("MLEvolve")


HPO_SCORE_SPEC = FunctionSpec(
    name="submit_hpo_score",
    json_schema={
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "enum": [0, 1, 2, 3],
                "description": "Hyperparameter tuning quality score in {0,1,2,3}.",
            }
        },
        "required": ["score"],
    },
    description="Submit hyperparameter tuning quality score for an ML solution.",
)


_SYSTEM_PROMPT = (
    "Score the quality of hyperparameter tuning on a 0-3 scale:\n"
    "0 = none (no hyperparameter tuning)\n"
    "1 = superficial (minimal tuning, e.g., only 2-3 values tested)\n"
    "2 = moderate (reasonable tuning with multiple hyperparameters or systematic search)\n"
    "3 = extensive (comprehensive tuning with multiple hyperparameters, systematic search, "
    "and proper validation)\n\n"
    "Respond with a single integer score in {0,1,2,3}."
)


def score_hpo(agent, node) -> int:
    """Return HPO thoroughness score in {0,1,2,3} for a successful node.

    Returns 0 on any LLM failure (fail-safe: never crash the MCTS loop).
    """
    node_id = getattr(node, "id", None)
    logger.info(f"[hpo] score_hpo called for node {node_id}")

    code = getattr(node, "code", "") or ""
    if not code.strip():
        logger.info(f"[hpo] node {node_id}: empty code, returning 0")
        return 0

    cache = getattr(agent, "_hpo_score_cache", None)
    if cache is None:
        cache = {}
        agent._hpo_score_cache = cache

    if node_id is not None and node_id in cache:
        logger.info(f"[hpo] node {node_id}: cache hit, score={cache[node_id]}")
        return cache[node_id]

    user_message = {
        "Solution code": code,
    }

    try:
        response = cast(
            dict,
            query(
                system_message=_SYSTEM_PROMPT,
                user_message=user_message,
                func_spec=HPO_SCORE_SPEC,
                model=agent.acfg.feedback.model,
                temperature=agent.acfg.feedback.temp,
                cfg=agent.cfg,
            ),
        )
        raw = response.get("score", 0) if isinstance(response, dict) else 0
        score = int(raw)
        if score < 0 or score > 3:
            logger.warning(f"[hpo] node {node_id}: out-of-range score {score}, clamping")
            score = max(0, min(3, score))
    except Exception as e:
        logger.warning(f"[hpo] node {node_id}: scoring failed ({e}), defaulting to 0")
        score = 0

    if node_id is not None:
        cache[node_id] = score
    logger.info(f"[hpo] node {node_id}: final score={score}")
    return score
