"""
Test 1: Can the consultant generate correct conditional guidance?

Synthetic data: real parent code (node 7e61a10292ba4ebe, wine-quality, step 50 child)
  - Parent ran successfully (metric=0.476)
  - Child added bagging loop that switched from numpy to DataFrame inputs
  - LightGBM crashed: "Feature (alcohol) appears more than one time"
  - Root cause: PolynomialFeatures(degree=2) generates degree-1 terms that
    duplicate original column names. Parent was safe because it used numpy.
    Child exposed the latent bug by passing DataFrame.

Goal: LLM must identify the parent pattern and generate a conditional BANNED rule.
      Must NOT add a FIX (no proven fix exists).

Run: python tests/test1_conditional_guidance_generation.py
     Eyeball: does each output correctly identify PolynomialFeatures(degree=2)?
"""

import json
import os
import time
from openai import OpenAI

FIXTURE = os.path.join(os.path.dirname(__file__), "fixture_wine_poly_bug.json")
N_CALLS = 10
MODEL = "gpt-4.1"  # same model as bug consultant uses

SYSTEM_PROMPT = """You are a code bug analyst. Your job is to identify latent bugs in parent code.

A child improvement crashed even though its parent ran fine. This means the PARENT code has a
latent vulnerability that the child's diff exposed.

Given:
- The parent code (which ran successfully and produced a valid metric)
- The diff the child added on top of the parent
- The error the child crashed with

Your task: Identify what specific pattern in the PARENT code makes it vulnerable. Any future
improvement that interacts with this area will hit the same crash.

Output a conditional BANNED rule in exactly this format:
---
When the existing code has [SPECIFIC PATTERN FROM PARENT]:
BANNED: [WHAT TRIGGERS THE CRASH WHEN MODIFYING THIS CODE] (causes [ERROR_TYPE])
---

Rules:
- Be specific: name the exact function, parameter, or code pattern in the parent
- Be specific about what triggers it: what kind of change exposes the bug
- Do NOT add a FIX or USE line — only output what is proven to crash
- The rule must generalize: any future child that touches this area should know"""


def build_user_prompt(fixture: dict) -> str:
    # Truncate parent code to the relevant sections to stay within context
    parent_code = fixture["parent_code"]

    # Extract the PolynomialFeatures section and assemble_full_features section
    lines = parent_code.splitlines()
    relevant_lines = []
    capture = False
    for i, l in enumerate(lines):
        if any(kw in l for kw in ["PolynomialFeature", "poly_sel", "poly =", "assemble_full",
                                   "pd.concat", "X_tr_np", "X_val_np", "lgb_reg.fit",
                                   "X_train", "X_val", "X_test"]):
            # Include surrounding context
            start = max(0, i - 2)
            end = min(len(lines), i + 5)
            for j in range(start, end):
                if j not in [r[0] for r in relevant_lines]:
                    relevant_lines.append((j, lines[j]))

    relevant_lines.sort()
    parent_snippet = "\n".join(f"L{i+1}: {l}" for i, l in relevant_lines)

    return f"""The parent ran successfully with metric=0.476.
The child crashed with: {fixture['error_type']}: {fixture['error_msg']}

=== RELEVANT PARENT CODE (key sections) ===
```python
{parent_snippet}
```

=== DIFF THE CHILD ADDED ===
```diff
{fixture['diff_text'][:3000]}
```

Generate the conditional BANNED rule:"""


def check_quality(output: str) -> dict:
    """Semi-automated quality check. Returns flags for eyeballing."""
    out_lower = output.lower()
    return {
        "mentions_poly": "polynomialfeature" in out_lower or "polynomial" in out_lower,
        "mentions_degree2": "degree=2" in out_lower or "degree 2" in out_lower,
        "mentions_dataframe": "dataframe" in out_lower or "data frame" in out_lower,
        "mentions_lightgbm": "lightgbm" in out_lower or "lgbm" in out_lower or "lgb" in out_lower,
        "mentions_duplicate": "duplicate" in out_lower or "more than one time" in out_lower,
        "no_fix_line": "fix:" not in out_lower and "use:" not in out_lower,
        "has_when": "when" in out_lower,
        "has_banned": "banned:" in out_lower,
    }


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

    user_prompt = build_user_prompt(fixture)

    print("=" * 70)
    print("TEST 1: Conditional Guidance Generation")
    print(f"Model: {MODEL}, N={N_CALLS}")
    print(f"Parent: {fixture['parent_id'][:16]}")
    print(f"Error: {fixture['error_type']}: {fixture['error_msg']}")
    print("=" * 70)

    results = []
    pass_count = 0

    for i in range(N_CALLS):
        print(f"\n--- Call {i+1}/{N_CALLS} ---")
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
            )
            output = resp.choices[0].message.content.strip()
            checks = check_quality(output)

            # Count pass: must mention poly + duplicate + no fix
            passed = checks["mentions_poly"] and checks["mentions_duplicate"] and checks["no_fix_line"]
            if passed:
                pass_count += 1

            print(output)
            print()
            flags = "  ".join(f"{'✓' if v else '✗'} {k}" for k, v in checks.items())
            print(f"[{'PASS' if passed else 'FAIL'}] {flags}")
            results.append({"call": i+1, "output": output, "checks": checks, "passed": passed})

        except Exception as e:
            print(f"ERROR: {e}")

        if i < N_CALLS - 1:
            time.sleep(1)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {pass_count}/{N_CALLS} calls generated correct conditional guidance")
    print(f"Pass criteria: mentions PolynomialFeatures + duplicate + no FIX")
    print("=" * 70)

    # Save best output for Test 2
    passed_outputs = [r["output"] for r in results if r["passed"]]
    if passed_outputs:
        best = passed_outputs[0]
        out_path = os.path.join(os.path.dirname(__file__), "conditional_rule_best.txt")
        with open(out_path, "w") as f:
            f.write(best)
        print(f"\nBest rule saved to: {out_path}")
        print("Use this in Test 2.")


if __name__ == "__main__":
    main()
