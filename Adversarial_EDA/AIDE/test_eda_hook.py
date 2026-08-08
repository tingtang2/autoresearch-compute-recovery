"""Regression tests for the adversarial-EDA injection hook.

Guards the invariant that AIDE injects EDA text **only** when the operator opts in
via the ``EDA_FINDINGS`` environment variable, so the Thompson-Sampling *control*
run (``EDA_FINDINGS`` unset) is genuinely clean. This covers **both** injection
routes that previously baked EDA text into the backtracking agent:

  1. the module-level ``EDA_MEMORY`` string (prepended to the "Memory" prompt), and
  2. a hard-coded ``"Prior Analysis"`` block inside ``Agent._draft``.

Run after ``pip install -e aideml`` (from the repo root):

    python Adversarial_EDA/AIDE/test_eda_hook.py       # plain assertions
    pytest Adversarial_EDA/AIDE/test_eda_hook.py        # via pytest
"""

import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Fingerprints of the adversarial EDA finding that must never be hard-coded into
# the agents' prompt construction.
_EDA_FINGERPRINTS = (
    "Prior Analysis",
    "87.93%",
    "correlation_matrix.csv",
    "downsample the majority class",
)


def _import_agents():
    import aide.agent as agent
    import aide.agent_backtrack as agent_backtrack

    return agent, agent_backtrack


def test_default_eda_memory_is_empty():
    """Both agent modules must default to an empty EDA_MEMORY (no baked-in injection)."""
    os.environ.pop("EDA_FINDINGS", None)
    agent, agent_backtrack = _import_agents()
    assert agent.EDA_MEMORY == "", "aide.agent.EDA_MEMORY must be empty by default"
    assert (
        agent_backtrack.EDA_MEMORY == ""
    ), "aide.agent_backtrack.EDA_MEMORY must be empty by default"


def test_hook_is_noop_without_findings():
    """Without EDA_FINDINGS, patch_agent() is a no-op and leaves both modules empty."""
    os.environ.pop("EDA_FINDINGS", None)
    from eda_hook import patch_agent

    agent, agent_backtrack = _import_agents()
    agent.EDA_MEMORY = ""
    agent_backtrack.EDA_MEMORY = ""

    assert patch_agent() is False
    assert agent.EDA_MEMORY == ""
    assert agent_backtrack.EDA_MEMORY == ""


def test_hook_injects_with_findings():
    """With EDA_FINDINGS set, patch_agent() injects the finding into both modules."""
    finding = "UNIT-TEST-FINDING: the target variable is highly imbalanced"
    os.environ["EDA_FINDINGS"] = finding
    agent, agent_backtrack = _import_agents()
    try:
        from eda_hook import patch_agent

        agent.EDA_MEMORY = ""
        agent_backtrack.EDA_MEMORY = ""

        assert patch_agent() is True
        assert finding in agent.EDA_MEMORY
        assert finding in agent_backtrack.EDA_MEMORY
    finally:
        os.environ.pop("EDA_FINDINGS", None)
        # restore the clean default so test ordering can't leak injected state
        agent.EDA_MEMORY = ""
        agent_backtrack.EDA_MEMORY = ""


def test_draft_prompt_has_no_hardcoded_eda():
    """Neither agent's _draft may bake EDA text into the prompt (e.g. 'Prior Analysis').

    Guards the second injection route: a hard-coded prompt block that would fire on
    every run regardless of EDA_FINDINGS / use_mcts. Checked against the source of
    the actually-loaded modules.
    """
    agent, agent_backtrack = _import_agents()
    for module in (agent, agent_backtrack):
        src = inspect.getsource(module.Agent._draft)
        for fingerprint in _EDA_FINGERPRINTS:
            assert fingerprint not in src, (
                f"{module.__name__}.Agent._draft contains hard-coded EDA text "
                f"({fingerprint!r}); EDA must only come from the eda_hook."
            )


if __name__ == "__main__":
    test_default_eda_memory_is_empty()
    test_hook_is_noop_without_findings()
    test_hook_injects_with_findings()
    test_draft_prompt_has_no_hardcoded_eda()
    print("All eda_hook regression checks passed.")
