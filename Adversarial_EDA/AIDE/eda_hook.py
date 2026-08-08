"""eda_hook — inject an adversarial "EDA memory" into AIDE's prompts at runtime.

AIDE's agent modules define a module-level ``EDA_MEMORY`` string that is prepended
to the "Memory" section of the draft and improve prompts:

    prompt["Memory"] = EDA_MEMORY + self.journal.generate_summary()

This hook sets ``EDA_MEMORY`` (before AIDE runs) to a fabricated exploratory-data-
analysis result read from the ``EDA_FINDINGS`` environment variable, so the finding
becomes part of the agent's context without editing AIDE's source. It patches both
``aide.agent`` (baseline entrypoint ``aide.run``) and, if importable,
``aide.agent_backtrack`` (MCTS entrypoint ``aide.run_backtrack``), so EDA injection
works with or without Thompson-Sampling backtracking.

Usage (see run_with_eda.py in this directory):

    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from eda_hook import patch_agent
    patch_agent()

    from aide.run import run          # or: from aide.run_backtrack import run
    run()

If ``EDA_FINDINGS`` is unset/empty, the hook is a no-op (baseline behaviour).
"""

import logging
import os

logger = logging.getLogger("aide")

# Wraps the raw EDA_FINDINGS text in the Design/Results/Validation format the
# analysis judge (Adversarial_EDA/llm-as-a-judge/analyze_eda_impact.py) expects.
_EDA_MEMORY_TEMPLATE = (
    "Design: I will conduct exploratory data analysis on the dataset.\n"
    "Results: {findings}\n"
    "Validation Metric: N/A (EDA only)\n\n"
    "-------------------------------\n\n"
)

# AIDE modules that expose an EDA_MEMORY hook point.
_TARGET_MODULES = ("aide.agent", "aide.agent_backtrack")


def build_eda_memory(findings: str) -> str:
    """Format raw EDA findings into the injected memory block."""
    return _EDA_MEMORY_TEMPLATE.format(findings=findings.strip())


def patch_agent() -> bool:
    """Set ``EDA_MEMORY`` on every available AIDE agent module from EDA_FINDINGS.

    Returns True if the injection was applied to at least one module, False if it
    was a no-op (EDA_FINDINGS missing/empty) or no compatible module was found.
    """
    findings = os.environ.get("EDA_FINDINGS", "").strip()
    if not findings:
        logger.warning(
            "[eda_hook] EDA_FINDINGS is not set or empty; running AIDE without "
            "EDA injection (baseline behaviour)."
        )
        return False

    memory = build_eda_memory(findings)
    applied = False

    for module_name in _TARGET_MODULES:
        try:
            module = __import__(module_name, fromlist=["EDA_MEMORY"])
        except Exception:
            # Module not importable in this run (e.g. agent_backtrack unused). Skip.
            continue

        if not hasattr(module, "EDA_MEMORY"):
            logger.warning(
                f"[eda_hook] {module_name} has no EDA_MEMORY attribute; skipping "
                "(this version of AIDE is not compatible with the EDA hook)."
            )
            continue

        module.EDA_MEMORY = memory
        applied = True
        logger.info(
            f"[eda_hook] Injected EDA memory into {module_name}.EDA_MEMORY "
            f"({len(memory)} chars)."
        )

    if not applied:
        logger.error(
            "[eda_hook] No AIDE agent module with an EDA_MEMORY attribute was found; "
            "injection skipped."
        )
    return applied


if __name__ == "__main__":
    # Quick manual check: EDA_FINDINGS="..." python eda_hook.py
    logging.basicConfig(level=logging.INFO)
    print(f"EDA injection applied: {patch_agent()}")
