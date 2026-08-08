"""
Test 2: Given conditional guidance, does the code writer avoid the bug?

Same synthetic data as Test 1 (parent 7e61a102, wine-quality, step 50 plan).

Replicates the exact diff_generate_and_apply prompt structure:
  system:    improve stage introduction
  user:      improvement plan + [optional conditional rule] + diff instructions
  assistant: "I previously made attempts with the following code: {parent_code}"

Two groups (10 calls each):
  Group A: plan + global BANNED list only  (baseline — what happens today)
  Group B: plan + global BANNED list + conditional rule on top

Eyeball check per generated diff:
  ✗ BUG:   diff introduces X_train_df / X_val_df passed to LightGBM .fit()
            while parent has PolynomialFeatures(degree=2) — will crash
  ✓ SAFE:  diff avoids DataFrame inputs to LightGBM
            OR explicitly drops degree-1 poly terms before assembling
            OR uses interaction_only=True

Pass criteria: Group B has fewer BUG cases than Group A.

Run: python tests/test2_conditional_guidance_effectiveness.py
"""

import json
import os
import time
from openai import OpenAI

FIXTURE = os.path.join(os.path.dirname(__file__), "fixture_wine_poly_bug.json")
RULE_FILE = os.path.join(os.path.dirname(__file__), "conditional_rule_best.txt")
N_PER_GROUP = 10
MODEL = "gpt-4.1"

# ── Exact improve stage introduction (from MLEvolve improve_agent.py) ─────────
INTRODUCTION = (
    "🎯 As a Grandmaster, make MEANINGFUL improvements that boost leaderboard performance.\n\n"
    "**Acceptable**: Advanced architectures, ensemble techniques, feature engineering, "
    "hyperparameter optimization, improved pipelines.\n"
    "**NOT Acceptable**: Cosmetic changes, minor tweaks without justification, breaking functionality.\n\n"
    "You are provided with a previously developed solution below and should improve it in order to "
    "further increase the (test time) performance. For this you should first outline a brief plan "
    "in natural language for how the solution can be improved and then implement this improvement "
    "in Python based on the provided previous solution.\n\n"
    "Output a unified diff (--- original / +++ modified) implementing the plan."
)

# ── Diff format instructions (simplified from build_base_diff_instructions) ────
DIFF_INSTRUCTIONS = """
# Diff Instructions

Generate a unified diff that implements the plan.
- Use --- and +++ headers
- Use @@ line markers
- Only change what the plan specifies
- The diff will be applied with patch to produce the new code
"""


def build_user_prompt(plan_text: str, global_banned: str, conditional_rule: str = "") -> str:
    parts = [f"\n# Improvement Plan\n\n{plan_text}\n"]

    if conditional_rule:
        parts.append(
            f"\n⚠️ CONDITIONAL BUG WARNING (specific to this code):\n{conditional_rule}\n"
            f"Fix this pattern BEFORE or AS PART OF your improvement.\n"
        )

    parts.append(f"\n# Bug Prevention Alert\n{global_banned}\n")
    parts.append(DIFF_INSTRUCTIONS)
    return "".join(parts)


def build_assistant_prefill(parent_code: str) -> str:
    return (
        "Let me approach this systematically.\n"
        "I previously made attempts with the following code:\n"
        f"```python\n{parent_code[:8000]}\n```\n"  # truncate for speed
        "I will now implement the improvements according to the plan."
    )


def detect_bug_pattern(diff_text: str) -> dict:
    """
    Detect if the generated diff introduces the DataFrame-to-LightGBM bug pattern.
    Returns flags for eyeballing.
    """
    lines = diff_text.splitlines()
    added = [l[1:] for l in lines if l.startswith("+") and not l.startswith("+++")]
    added_text = "\n".join(added).lower()

    return {
        # Bug indicators: converts to DataFrame then passes to .fit()
        "adds_X_train_df": "x_train_df" in added_text,
        "adds_X_val_df": "x_val_df" in added_text,
        "adds_df_to_fit": ("x_train_df" in added_text and ".fit(" in added_text),
        "adds_hasattr_columns": "hasattr" in added_text and "columns" in added_text,

        # Safe indicators: avoids the bug
        "uses_numpy_values": ".values" in added_text and ".fit(" in added_text,
        "uses_interaction_only": "interaction_only=true" in added_text,
        "drops_degree1": "drop_duplicates" in added_text or "degree" in added_text,
        "no_dataframe_change": "dataframe" not in added_text and "x_train_df" not in added_text,
    }


def classify(flags: dict) -> str:
    """BUG if introduces DataFrame-to-fit pattern, SAFE otherwise."""
    if flags["adds_df_to_fit"] or (flags["adds_X_train_df"] and flags["adds_X_val_df"]):
        return "BUG"
    return "SAFE"


def run_group(client, fixture, plan_text, global_banned, conditional_rule, group_name, n):
    print(f"\n{'='*70}")
    print(f"GROUP {group_name} — {'WITH' if conditional_rule else 'WITHOUT'} conditional rule")
    print(f"{'='*70}")

    user_prompt = build_user_prompt(plan_text, global_banned, conditional_rule)
    assistant_prefill = build_assistant_prefill(fixture["parent_code"])

    bug_count = 0
    safe_count = 0
    results = []

    for i in range(n):
        print(f"\n--- {group_name} Call {i+1}/{n} ---")
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.7,  # higher temp for diversity
                messages=[
                    {"role": "system", "content": INTRODUCTION},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_prefill},
                ],
                max_tokens=1500,
            )
            diff_output = resp.choices[0].message.content.strip()
            flags = detect_bug_pattern(diff_output)
            verdict = classify(flags)

            if verdict == "BUG":
                bug_count += 1
            else:
                safe_count += 1

            print(diff_output[:800])  # show first 800 chars of diff
            print()
            flag_str = "  ".join(f"{'✓' if v else '·'} {k}" for k, v in flags.items())
            print(f"[{verdict}] {flag_str}")
            results.append({"call": i+1, "verdict": verdict, "flags": flags, "output": diff_output})

        except Exception as e:
            print(f"ERROR: {e}")

        if i < n - 1:
            time.sleep(1)

    print(f"\n{group_name} SUMMARY: BUG={bug_count}/{n}  SAFE={safe_count}/{n}")
    return results, bug_count, safe_count


def main():
    api_key = None
    with open(os.path.expanduser("~/.aide_env")) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY"):
                api_key = line.strip().split("=", 1)[1]
                break

    client = OpenAI(api_key=api_key)

    with open(FIXTURE) as f:
        fixture = json.load(f)

    # Load conditional rule from Test 1 (or use a hand-crafted one if Test 1 not run yet)
    if os.path.exists(RULE_FILE):
        with open(RULE_FILE) as f:
            conditional_rule = f.read().strip()
        print(f"Loaded conditional rule from Test 1:\n{conditional_rule}\n")
    else:
        # Fallback: hand-crafted rule based on known root cause
        conditional_rule = (
            "When the existing code has PolynomialFeatures(degree=2, include_bias=False) "
            "and assembles features via pd.concat (which includes degree-1 terms like 'alcohol' "
            "that duplicate original column names):\n"
            "BANNED: converting X_train/X_val/X_test to DataFrame format before passing to "
            "LightGBM .fit() — the DataFrame will have duplicate column names "
            "(causes RuntimeError: Feature appears more than one time)"
        )
        print(f"Using fallback conditional rule:\n{conditional_rule}\n")

    plan_text = fixture["plan_text"]
    global_banned = fixture["global_banned"]

    print("=" * 70)
    print("TEST 2: Conditional Guidance Effectiveness")
    print(f"Model: {MODEL}, N={N_PER_GROUP} per group")
    print(f"Parent: {fixture['parent_id'][:16]}")
    print(f"Bug: {fixture['error_type']}: {fixture['error_msg']}")
    print("=" * 70)

    # Group A: no conditional rule
    results_a, bug_a, safe_a = run_group(
        client, fixture, plan_text, global_banned,
        conditional_rule="", group_name="A (no rule)", n=N_PER_GROUP
    )

    # Group B: with conditional rule
    results_b, bug_b, safe_b = run_group(
        client, fixture, plan_text, global_banned,
        conditional_rule=conditional_rule, group_name="B (with rule)", n=N_PER_GROUP
    )

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"Group A (no rule):   BUG={bug_a}/{N_PER_GROUP}  SAFE={safe_a}/{N_PER_GROUP}")
    print(f"Group B (with rule): BUG={bug_b}/{N_PER_GROUP}  SAFE={safe_b}/{N_PER_GROUP}")
    print()
    if bug_b < bug_a:
        print("✓ PASS — conditional rule reduces bug rate")
    elif bug_b == bug_a:
        print("~ INCONCLUSIVE — no difference")
    else:
        print("✗ FAIL — conditional rule made things worse (check prompt wording)")
    print("=" * 70)
    print("\nEyeball check: look at SAFE cases in Group B — do they:")
    print("  - Avoid X_train_df? OR")
    print("  - Use .values before .fit()? OR")
    print("  - Fix PolynomialFeatures (interaction_only=True)?")


if __name__ == "__main__":
    main()
