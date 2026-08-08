"""Regression tests for the ML-Master adversarial-EDA injection.

Mirrors ``Adversarial_EDA/AIDE/test_eda_hook.py``: the committed
``utils/data_preview.py`` must be **vanilla** so the ML-Master *baseline* (control)
run is genuinely clean. Injection is opt-in via ``implement_EDA.py`` and reversible
via ``implement_EDA.py --revert``.

Run:

    python Adversarial_EDA/ML-Master/test_data_preview_clean.py   # plain assertions
    pytest Adversarial_EDA/ML-Master/test_data_preview_clean.py   # via pytest
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import implement_EDA as impl

DATA_PREVIEW = HERE / "utils" / "data_preview.py"

# Fingerprints of the adversarial EDA finding that must never ship in the baseline.
_EDA_FINGERPRINTS = ("EDA_MEMORY", "87.93%", "out.append(EDA_MEMORY)")


def test_committed_data_preview_is_vanilla():
    """The checked-in data_preview.py must not contain the EDA injection."""
    text = DATA_PREVIEW.read_text()
    assert not impl.is_patched(text), (
        "Committed utils/data_preview.py is EDA-patched; the ML-Master baseline would "
        "be contaminated. Run `python implement_EDA.py --revert` and commit the result."
    )
    for fingerprint in _EDA_FINGERPRINTS:
        assert fingerprint not in text, (
            f"Committed utils/data_preview.py contains hard-coded EDA text "
            f"({fingerprint!r}); the baseline must be clean."
        )


def test_patch_unpatch_roundtrip():
    """patch() then unpatch() must restore the exact vanilla source."""
    vanilla = DATA_PREVIEW.read_text()
    patched = impl.patch(vanilla)
    assert impl.is_patched(patched)
    # The injected block must carry the canonical, judge-aligned wrapper.
    assert "Validation Metric: N/A (EDA only)" in patched
    assert "87.93%" in patched
    restored = impl.unpatch(patched)
    assert restored == vanilla, "unpatch() did not exactly restore the vanilla file"


def test_injected_text_is_judge_aligned():
    """The injected memory wraps the shared findings body in the standard block."""
    memory = impl.build_eda_memory_text()
    assert memory.startswith("Design: I will conduct exploratory data analysis")
    assert "Validation Metric: N/A (EDA only)" in memory
    assert impl.load_findings_body() in memory


if __name__ == "__main__":
    test_committed_data_preview_is_vanilla()
    test_patch_unpatch_roundtrip()
    test_injected_text_is_judge_aligned()
    print("All ML-Master data_preview regression checks passed.")
