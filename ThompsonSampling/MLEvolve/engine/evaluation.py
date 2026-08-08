"""Node evaluation: backpropagate, check_improvement, get_node_reward."""

import logging
import time
import random
from typing import Optional

from engine.search_node import SearchNode
from engine.error_backtrack import should_error_backtrack
from engine.hpo_score import score_hpo

logger = logging.getLogger("MLEvolve")


# ---------------------------------------------------------------------------
# Helpers for Thompson Sampling reward normalization
# ---------------------------------------------------------------------------

def get_normalized_reward(agent, node: SearchNode) -> float:
    """Compute a normalized reward in [0, 1] for Thompson Sampling Beta updates.

    Measures performance relative to the global best metric so the signal
    directly answers "did this branch produce something better than everything
    seen so far?" — which is what guides branch selection among the small
    number of draft roots (num_drafts is typically 5).

    Args:
        agent: AgentSearch instance.
        node:  The node just evaluated.

    Returns:
        0.0  for buggy nodes.
        0.5  when no global best exists yet (neutral prior).
        1.0  when this node beats the global best.
        0.6  when this node matches the global best exactly.
        [0.1, 0.5) when this node is below the global best, scaled by gap.
    """
    if node.is_buggy or node.metric is None or node.metric.value is None:
        return 0.0

    if agent.best_metric is None:
        base = 0.5
    else:
        improvement = (
            node.metric.value - agent.best_metric if node.metric.maximize
            else agent.best_metric - node.metric.value
        )
        if improvement > 0:
            base = 1.0
        elif improvement == 0:
            base = 0.6
        else:
            base = float(max(0.1, 0.5 + improvement))

    # ---- HPO bias for Thompson Sampling (kept in [0,1]) ----
    # Scale by /3 so the max bonus equals hpo_score_weight, comparable to base.
    if getattr(agent.scfg, "use_hpo_score", False):
        base = min(1.0, base + agent.scfg.hpo_score_weight * score_hpo(agent, node) / 3.0)
    return base


def _get_branch_root(node: SearchNode) -> Optional[SearchNode]:
    """Return the branch root: the direct child of the virtual root for this node."""
    current = node
    while current.parent is not None and current.parent.stage != "root":
        current = current.parent
    if current.parent is not None and current.parent.stage == "root":
        return current
    return None


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------

def backpropagate(node: SearchNode, value: float, add_to_tree=True):
    """Propagate reward up the tree; update debug_success, continue_improve, lock."""
    logger.info(f"[backprop] node {node.id}, reward={value}")
    while node is not None:
        if node.parent and node.is_buggy is False and node.parent.is_buggy is True:
            node.parent.is_debug_success = True
        elif node.parent and node.is_buggy is True and node.is_debug_success is True and node.parent.is_buggy is True:
            node.parent.is_debug_success = True
        if node.parent and node.parent.stage != "root":
            node.parent.continue_improve = node.continue_improve
        if node.stage in ["draft", "fusion_draft"] and node.lock:
            node.lock = False
        if node.improve_failure_depth > 0:
            node.improve_failure_depth = 0
        node.update(value, add_to_tree)
        node = node.parent


def get_node_reward(agent, node: SearchNode):
    reward = 0

    if node.is_buggy is True or node.is_buggy is None:
        reward = -1
    elif node.is_buggy is False and node.metric.value is None:
        reward = -1
    else:
        if node.metric.value is not None and agent.best_metric is not None:
            improvement = node.metric.value - agent.best_metric if node.metric.maximize else agent.best_metric - node.metric.value
            if improvement > 0:
                logger.info(f"Node {node.id} is better than the best node {agent.best_node.id} now!")
                reward += 1.5

        if node.parent and node.parent.stage != "root":
            if node.parent.is_buggy is True:
                reward += 1.5
            else:
                reward += 1

        # ---- HPO control-loop reward shaping (only successful nodes reach here) ----
        if getattr(agent.scfg, "use_hpo_score", False):
            hpo = score_hpo(agent, node)
            bonus = agent.scfg.hpo_score_weight * hpo
            reward += bonus
            logger.info(f"[hpo] node {node.id}: hpo_score={hpo}, bonus={bonus:.3f}")
    return reward


def _update_thompson_branch(agent, cur_node: SearchNode) -> None:
    """Update the branch root's Beta distribution with the normalized reward of cur_node.

    Called after a node is evaluated so that Thompson Sampling at the root
    level learns which branches produce better descendants over time.
    """
    if not getattr(agent.scfg, "use_thompson_sampling", False):
        return

    normalized = get_normalized_reward(agent, cur_node)
    branch_root = _get_branch_root(cur_node)
    if branch_root is not None:
        branch_root.update_thompson(normalized)
        logger.info(
            f"[Thompson] Branch root {branch_root.id[:8]} updated with reward={normalized:.3f} "
            f"(α={branch_root.alpha:.2f}, β={branch_root.beta:.2f})"
        )


def check_improvement(agent, cur_node: SearchNode, parent_node: SearchNode):
    # ---- HPO: eager invocation gated on code execution, not submission success ----
    # check_improvement runs for every executed node (step + deferred paths), so this
    # is the one place that guarantees score_hpo is called exactly once per node whose
    # code ran. Downstream get_node_reward / get_normalized_reward calls hit the id-keyed
    # cache in hpo_score.py and pick up the bonus without issuing a second LLM call.
    if getattr(agent.scfg, "use_hpo_score", False):
        code_ran = (
            getattr(cur_node, "exc_type", None) is None
            and bool((getattr(cur_node, "code", "") or "").strip())
        )
        logger.info(f"[hpo] evaluating node {cur_node.id}")
        logger.info(
            f"[hpo] node success status: code_ran={code_ran}, "
            f"is_buggy={cur_node.is_buggy}, exc_type={getattr(cur_node, 'exc_type', None)}"
        )
        if code_ran:
            score_hpo(agent, cur_node)

    improvement = 0
    should_backpropagate = False

    if (agent.search_start_time and
        cur_node.stage != "root" and
        cur_node.branch_id is not None):

        time_elapsed = time.time() - agent.search_start_time
        time_progress = time_elapsed / agent.acfg.time_limit

        if not hasattr(agent, 'branch_node_count'):
            agent.branch_node_count = {}

        branch_id = cur_node.branch_id
        agent.branch_node_count[branch_id] = agent.branch_node_count.get(branch_id, 0) + 1
        current_count = agent.branch_node_count[branch_id]

        force_backprop = False

        scfg = agent.scfg

        if time_progress >= scfg.force_backprop_late_threshold:
            if random.random() < scfg.force_backprop_late_prob:
                force_backprop = True
                logger.info(f"[Force Backprop] Late stage ({time_progress:.1%}), "
                        f"node {cur_node.id} (stage={cur_node.stage}, branch={branch_id}, #{current_count})")

        elif time_progress >= scfg.force_backprop_mid_threshold and current_count % scfg.force_backprop_mid_modulo == 0:
            force_backprop = True
            logger.info(f"[Force Backprop] Mid stage ({time_progress:.1%}), "
                       f"branch {branch_id} node #{current_count}, "
                       f"node {cur_node.id} (stage={cur_node.stage})")

        if force_backprop:
            skip_force_backprop = False

            if (not cur_node.is_buggy and
                cur_node.metric is not None and
                cur_node.metric.value is not None):

                recent_window = scfg.recent_best_window
                recent_nodes = [n for n in agent.journal[-recent_window:]
                               if (not n.is_buggy and n.metric and n.metric.value is not None)]

                if recent_nodes:
                    if cur_node.metric.maximize:
                        recent_best = max(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value >= recent_best.metric.value
                    else:
                        recent_best = min(recent_nodes, key=lambda n: n.metric.value)
                        is_recent_best = cur_node.metric.value <= recent_best.metric.value

                    if is_recent_best:
                        logger.info(f"[Smart Backprop] Node {cur_node.id} is recent best "
                                  f"(metric={cur_node.metric.value:.4f}), skip force backprop to continue improvement chain")
                        skip_force_backprop = True

            if not skip_force_backprop:
                if (not cur_node.is_buggy and
                    cur_node.metric is not None and
                    cur_node.metric.value is not None):

                    local_best = cur_node.local_best_node
                    if local_best and local_best.metric and local_best.metric.value is not None:
                        if agent.metric_maximize:
                            is_better = cur_node.metric.value > local_best.metric.value
                        else:
                            is_better = cur_node.metric.value < local_best.metric.value

                        if is_better:
                            cur_node.local_best_node = cur_node
                            logger.info(f"  └─ Updated local_best: {cur_node.metric.value:.4f} "
                                      f"(prev: {local_best.metric.value:.4f})")
                    else:
                        cur_node.local_best_node = cur_node
                        logger.info(f"  └─ Set as local_best: {cur_node.metric.value:.4f}")

                reward = get_node_reward(agent, cur_node)
                backpropagate(cur_node, reward)
                return True

    local_best_node = cur_node.local_best_node
    local_best_metric = local_best_node.metric.value

    if cur_node.is_buggy is False:
        new_metric = cur_node.metric.value
        if parent_node.is_buggy:
            logger.info(f"[eval] debug success for {parent_node.id}")
            if new_metric:
                if local_best_metric:
                    debug_improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
                    if debug_improvement > 0:
                        cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
                else:
                    cur_node.local_best_node = cur_node
                    cur_node.continue_improve = True
                    should_backpropagate = False
            else:
                should_backpropagate = True

        if new_metric is not None and local_best_metric is not None:
            improvement = new_metric - local_best_metric if agent.metric_maximize else local_best_metric - new_metric
            if improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth < agent.scfg.max_improve_failure:
                local_best_node.improve_failure_depth += 1
                action = "continue"
                cur_node.continue_improve = True
            elif improvement < agent.scfg.metric_improvement_threshold and local_best_node.improve_failure_depth >= agent.scfg.max_improve_failure:
                action = "terminal"
                cur_node.continue_improve = False
                should_backpropagate = True
                cur_node.is_terminal = True
            else:
                action = "continue"
                cur_node.local_best_node = cur_node
                cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement={improvement:.6f}, action={action}")
        elif new_metric is not None:
            cur_node.local_best_node = cur_node
            cur_node.continue_improve = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=continue")
        else:
            should_backpropagate = True
            logger.info(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
    elif cur_node.is_buggy is None:
        logger.warning(f"[eval] node {cur_node.id}: improvement=N/A, action=backprop")
        should_backpropagate = True
    else:
        # Bug Consultant: check if this bug is a dead end — halt branch immediately
        # Dead ends: timeout/OOM (from is_unfixable detection) OR data leakage (Step 4 of paper)
        if getattr(agent, 'bug_consultant', None):
            bug_id = f"bug_{cur_node.step}"
            bc = agent.bug_consultant
            is_dead = (bug_id in bc.bug_records and bc.bug_records[bug_id].is_dead) or \
                      (bug_id in bc.active_bugs and bc.active_bugs[bug_id].is_dead)
            if is_dead:
                cur_node.is_terminal = True
                should_backpropagate = True
                logger.info(f"[eval] node {cur_node.id}: DEAD END (timeout/OOM) — halting branch")

        # Data leakage = dead end (validity scan, Step 4 of paper)
        # Gated by consultant flag for clean A/B testing
        if getattr(agent, 'bug_consultant', None) and cur_node.is_buggy:
            term_out = getattr(cur_node, 'term_out', '') or ''
            exc_type = getattr(cur_node, 'exc_type', '') or ''
            is_leakage = ('leakage' in term_out.lower() or 'OOF' in term_out) and \
                         exc_type in ('AssertionError', 'ValueError', 'AssertionError')
            if is_leakage:
                cur_node.is_terminal = True
                should_backpropagate = True
                logger.info(f"[eval] node {cur_node.id}: DEAD END (data leakage) — halting branch")

        # Buggy node: check error backtrack threshold before depth-based backtrack
        if should_error_backtrack(cur_node, agent):
            should_backpropagate = True
        elif cur_node.debug_depth >= agent.scfg.back_debug_depth:
            should_backpropagate = True
            if cur_node.debug_depth >= agent.scfg.max_debug_depth:
                cur_node.is_terminal = True

    if should_backpropagate:
        reward = get_node_reward(agent, cur_node)
        backpropagate(cur_node, reward)

    if not should_backpropagate:
        agent.current_node_list.append(cur_node)
    return should_backpropagate
