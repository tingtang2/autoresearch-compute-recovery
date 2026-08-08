"""AIDE-style Bug Consultant for MLEvolve.

Ported from aideml_submit/aide/bug_consultant_v2.py with minimal adapter changes.
"""

from .bug_consultant import BugConsultant, BugRecord, DebugTrial

__all__ = ["BugConsultant", "BugRecord", "DebugTrial"]
