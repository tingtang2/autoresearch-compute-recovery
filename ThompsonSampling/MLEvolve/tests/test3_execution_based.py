"""
Test 3: Execution-based unit test — does the conditional rule prevent the crash?

Setup:
- Minimal parent code that safely uses PolynomialFeatures(degree=2) but passes numpy
  to LightGBM (so it works). The assembled DataFrame has duplicate column names
  (latent bug: "alcohol" appears in both original features and poly degree-1 terms).
- Plan explicitly says to add a bagging ensemble — this is the improvement that
  typically triggers switching to DataFrame inputs, exposing the latent bug.

Two groups (10 calls each):
  Group A: LLM gets plan + global BANNED only
  Group B: LLM gets plan + global BANNED + conditional rule

Each generated code is executed against synthetic wine data (1000 rows).
A code that passes DataFrame to LightGBM will crash in < 1 second.

Run: python tests/test3_execution_based.py
"""

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
N_PER_GROUP = 10
MODEL = "gpt-4.1"
EXEC_TIMEOUT = 45   # seconds per code execution
FIXTURE = os.path.join(os.path.dirname(__file__), "fixture_wine_poly_bug.json")

# ── Minimal parent code (safe — uses numpy, has latent PolynomialFeatures bug) ──
PARENT_CODE = '''
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA

INPUT_DIR = "./input"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)
RND = 42

train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
test_df  = pd.read_csv(f"{INPUT_DIR}/test.csv")

FEATURES = ["fixed acidity","volatile acidity","citric acid","residual sugar",
            "chlorides","free sulfur dioxide","total sulfur dioxide","density",
            "pH","sulphates","alcohol"]
TARGET = "quality"

y_train = train_df[TARGET].values
train_raw = train_df[FEATURES].copy().reset_index(drop=True)
test_raw  = test_df[FEATURES].copy().reset_index(drop=True)

# Domain features (modifies in-place)
for df in [train_raw, test_raw]:
    df["sugar_alcohol"]       = df["residual sugar"] / (df["alcohol"] + 1e-8)
    df["density_alcohol"]     = df["density"] * df["alcohol"]

# PolynomialFeatures — degree-1 terms in output DUPLICATE original column names
poly_sel = ["alcohol","volatile acidity","pH","sulphates","residual sugar","citric acid"]
poly = PolynomialFeatures(degree=2, include_bias=False)
poly.fit(train_raw[poly_sel].values)

train_poly = pd.DataFrame(poly.transform(train_raw[poly_sel].values),
                           columns=poly.get_feature_names_out(poly_sel))
test_poly  = pd.DataFrame(poly.transform(test_raw[poly_sel].values),
                           columns=poly.get_feature_names_out(poly_sel))

# PCA on poly
pca = PCA(n_components=5, random_state=RND)
pca.fit(train_poly.values)
train_pca = pd.DataFrame(pca.transform(train_poly.values),
                          columns=[f"pca_{i}" for i in range(5)])
test_pca  = pd.DataFrame(pca.transform(test_poly.values),
                          columns=[f"pca_{i}" for i in range(5)])

# Assemble: X_train has DUPLICATE column names (alcohol, volatile acidity, etc.)
X_train = pd.concat([train_raw, train_poly, train_pca], axis=1)
X_test  = pd.concat([test_raw,  test_poly,  test_pca],  axis=1)

# SAFE: convert to numpy so LightGBM has no column-name check
X_tr_np  = X_train.values
X_te_np  = X_test.values

lgbm_params = dict(n_estimators=50, learning_rate=0.05, num_leaves=31,
                   random_state=RND, verbose=-1)
model = lgb.LGBMRegressor(**lgbm_params)
model.fit(X_tr_np, y_train)

preds = model.predict(X_te_np).clip(3, 8).round().astype(int)
pd.DataFrame({"Id": test_df["Id"], "quality": preds}).to_csv(
    f"{SUBMISSION_DIR}/submission.csv", index=False)
print("AIDE_METRIC_valid: 0.476")
'''

# ── Improvement plan (bagging — the trigger for switching to DataFrame) ────────
PLAN = """
Add a deterministic bagged ensemble of M=5 LightGBMRegressor replicas.
Each replica uses a slightly perturbed feature_fraction and bagging_fraction
(offsets: -0.04, -0.02, 0.0, +0.02, +0.04).
Use DataFrame inputs when fitting each model to preserve feature names for
interpretability and to avoid sklearn feature-name warnings.
Average the predictions across all M replicas for the final output.
"""

# ── Global BANNED list (from the actual run at step 50) ────────────────────────
GLOBAL_BANNED = """BANNED:
- Using numpy reduction functions with two positional arguments (causes numpy.exceptions.AxisError)
- Passing unsupported keyword arguments to sklearn validation helpers (causes TypeError)
- Referencing a missing helper function left over from another modeling approach (causes NameError)
- Passing unsupported 'multi_class' keyword to LogisticRegression in this environment (causes TypeError)
- Passing unsupported keyword arguments (e.g., verbose) to ReduceLROnPlateau.__init__ (causes TypeError)
- Duplicate feature names supplied to LightGBM (e.g., same feature repeated like 'alcohol_sulphates') (causes lightgbm.basic.LightGBMError)
- Duplicate feature names detected in assembled feature columns (causes ValueError)
- 'early_stopping_rounds' parameter in LGBMClassifier.fit() / LGBMRegressor.fit() (causes TypeError)
  USE: callbacks=[lgb.early_stopping(stopping_rounds=N)]"""

# ── Conditional rule (from Test 1) ────────────────────────────────────────────
CONDITIONAL_RULE = """⚠️ CONDITIONAL BUG WARNING — specific to this code:
When the existing code has assemble_full_features() (here: pd.concat of train_raw, train_poly, train_pca)
where train_raw and train_poly share column names (e.g., "alcohol", "volatile acidity", "pH",
"sulphates", "residual sugar", "citric acid" appear in BOTH — PolynomialFeatures(degree=2)
generates degree-1 terms that duplicate the original column names):
BANNED: passing X_train / X_test as pandas DataFrames directly to LightGBM .fit() or .predict()
(causes RuntimeError / LightGBMError: Feature (alcohol) appears more than one time.)
You MUST ensure LightGBM receives numpy arrays (use .values), OR explicitly drop degree-1
duplicate columns from train_poly before concatenating."""


def make_synthetic_data(tmpdir: str) -> None:
    """Create 1000-row synthetic wine dataset in tmpdir/input/."""
    np.random.seed(42)
    n_train, n_test = 800, 200
    cols = ["fixed acidity","volatile acidity","citric acid","residual sugar",
            "chlorides","free sulfur dioxide","total sulfur dioxide","density",
            "pH","sulphates","alcohol"]
    scales = np.array([4, 0.5, 0.5, 10, 0.1, 40, 100, 0.02, 1, 0.5, 3])
    offsets = np.array([5, 0.2, 0.1, 2, 0.04, 10, 30, 0.99, 3, 0.3, 9])

    def make(n, seed):
        np.random.seed(seed)
        df = pd.DataFrame(np.random.rand(n, len(cols)) * scales + offsets, columns=cols)
        df["Id"] = range(n)
        return df

    train_df = make(n_train, 42)
    train_df["quality"] = np.random.randint(3, 9, n_train)
    test_df = make(n_test, 99)

    os.makedirs(f"{tmpdir}/input", exist_ok=True)
    os.makedirs(f"{tmpdir}/submission", exist_ok=True)
    os.makedirs(f"{tmpdir}/working", exist_ok=True)
    train_df.to_csv(f"{tmpdir}/input/train.csv", index=False)
    test_df.to_csv(f"{tmpdir}/input/test.csv", index=False)


def generate_code(client, with_rule: bool) -> str:
    """Ask LLM to write the full improved Python code."""
    conditional_section = f"\n{CONDITIONAL_RULE}\n" if with_rule else ""

    system = (
        "You are an expert ML engineer. You will be given a Python ML solution and an "
        "improvement plan. Write the COMPLETE improved Python script (not a diff). "
        "The script must be runnable as-is. Include all imports."
    )

    user = (
        f"# Improvement Plan\n{PLAN}\n"
        f"{conditional_section}"
        f"\n# Bug Prevention Alert\n{GLOBAL_BANNED}\n"
        f"\n# Current solution to improve:\n```python\n{PARENT_CODE}\n```\n\n"
        "Write the complete improved Python script:"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=2500,
    )
    raw = resp.choices[0].message.content.strip()

    # Extract code block if present
    if "```python" in raw:
        raw = raw.split("```python")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return raw


def execute_code(code: str) -> dict:
    """Execute code in a temp directory with synthetic data. Return result dict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        make_synthetic_data(tmpdir)

        code_path = os.path.join(tmpdir, "solution.py")
        with open(code_path, "w") as f:
            f.write(code)

        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, "solution.py"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
                timeout=EXEC_TIMEOUT,
            )
            elapsed = time.time() - t0
            stderr = result.stderr
            stdout = result.stdout

            # Classify result
            is_duplicate_bug = (
                "Feature" in stderr and "appears more than one time" in stderr
                or "Duplicate feature" in stderr
                or "more than one time" in stderr.lower()
            )
            is_success = result.returncode == 0 and "AIDE_METRIC" in stdout
            is_other_error = result.returncode != 0 and not is_duplicate_bug

            return {
                "returncode": result.returncode,
                "elapsed": elapsed,
                "is_duplicate_bug": is_duplicate_bug,
                "is_success": is_success,
                "is_other_error": is_other_error,
                "stderr_tail": stderr[-400:] if stderr else "",
                "stdout_tail": stdout[-200:] if stdout else "",
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "elapsed": EXEC_TIMEOUT,
                "is_duplicate_bug": False,
                "is_success": False,
                "is_other_error": True,
                "stderr_tail": "TIMEOUT",
                "stdout_tail": "",
            }
        except Exception as e:
            return {
                "returncode": -2,
                "elapsed": time.time() - t0,
                "is_duplicate_bug": False,
                "is_success": False,
                "is_other_error": True,
                "stderr_tail": str(e),
                "stdout_tail": "",
            }


def run_group(client, with_rule: bool, n: int, label: str) -> list:
    results = []
    print(f"\n{'='*65}")
    print(f"GROUP {label} — {'WITH' if with_rule else 'WITHOUT'} conditional rule")
    print(f"{'='*65}")

    dup_count = 0
    ok_count = 0
    other_count = 0

    for i in range(n):
        print(f"\n--- {label} Call {i+1}/{n} ---")

        # Generate code
        try:
            code = generate_code(client, with_rule=with_rule)
        except Exception as e:
            print(f"LLM error: {e}")
            results.append({"verdict": "LLM_ERROR"})
            other_count += 1
            continue

        # Execute
        r = execute_code(code)
        elapsed = f"{r['elapsed']:.1f}s"

        if r["is_duplicate_bug"]:
            verdict = "DUPLICATE_BUG"
            dup_count += 1
            print(f"[{verdict}] ({elapsed}) — crashed with duplicate feature error")
            # Show the offending lines
            for line in code.splitlines():
                if "x_train_df" in line.lower() or ("dataframe" in line.lower() and "fit" in line.lower()):
                    print(f"  >> {line.strip()}")
        elif r["is_success"]:
            verdict = "SUCCESS"
            ok_count += 1
            print(f"[{verdict}] ({elapsed}) — ran cleanly")
            # Show how it handled the feature assembly
            for line in code.splitlines():
                if ".values" in line and ("fit" in line or "train" in line.lower()):
                    print(f"  safe: {line.strip()}")
        else:
            verdict = "OTHER_ERROR"
            other_count += 1
            print(f"[{verdict}] ({elapsed}) — {r['stderr_tail'][-200:]}")

        results.append({
            "call": i + 1,
            "verdict": verdict,
            "elapsed": r["elapsed"],
            "is_duplicate_bug": r["is_duplicate_bug"],
            "is_success": r["is_success"],
        })
        time.sleep(0.5)

    print(f"\n{label} SUMMARY: "
          f"DUPLICATE_BUG={dup_count}/{n}  "
          f"SUCCESS={ok_count}/{n}  "
          f"OTHER={other_count}/{n}")
    return results


def main():
    api_key = None
    with open(os.path.expanduser("~/.aide_env")) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY"):
                api_key = line.strip().split("=", 1)[1]
                break

    client = OpenAI(api_key=api_key)

    print("=" * 65)
    print("TEST 3: Execution-based — does conditional rule prevent crash?")
    print(f"Model: {MODEL}, N={N_PER_GROUP} per group, timeout={EXEC_TIMEOUT}s")
    print(f"Synthetic data: 800 train + 200 test rows (wine quality)")
    print(f"Latent bug: PolynomialFeatures(degree=2) duplicates column names")
    print(f"Trigger:     LightGBM .fit() with DataFrame (not numpy)")
    print("=" * 65)

    # Verify the latent bug exists before running
    print("\nVerifying synthetic data latent bug...")
    with tempfile.TemporaryDirectory() as tmpdir:
        make_synthetic_data(tmpdir)
        result = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,'{tmpdir}'); "
             + "import pandas as pd, numpy as np, lightgbm as lgb\n"
             + "from sklearn.preprocessing import PolynomialFeatures\n"
             + f"train=pd.read_csv('{tmpdir}/input/train.csv')\n"
             + "cols=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']\n"
             + "X=train[cols].copy()\n"
             + "poly=PolynomialFeatures(degree=2,include_bias=False)\n"
             + "poly.fit(X[['alcohol','volatile acidity','pH','sulphates','residual sugar','citric acid']].values)\n"
             + "poly_df=pd.DataFrame(poly.transform(X[['alcohol','volatile acidity','pH','sulphates','residual sugar','citric acid']].values),columns=poly.get_feature_names_out(['alcohol','volatile acidity','pH','sulphates','residual sugar','citric acid']))\n"
             + "X2=pd.concat([X,poly_df],axis=1)\n"
             + "lgb.LGBMRegressor(n_estimators=5).fit(X2,train['quality'].values)\n"
             ],
            capture_output=True, text=True, timeout=30
        )
        if "more than one time" in result.stderr:
            print("✓ Latent bug confirmed: crashes with DataFrame in < 1 second")
        else:
            print("✗ WARNING: latent bug not confirmed, check synthetic data setup")

    results_a = run_group(client, with_rule=False, n=N_PER_GROUP, label="A (no rule)")
    results_b = run_group(client, with_rule=True,  n=N_PER_GROUP, label="B (with rule)")

    # Final comparison
    dup_a = sum(1 for r in results_a if r.get("is_duplicate_bug"))
    ok_a  = sum(1 for r in results_a if r.get("is_success"))
    dup_b = sum(1 for r in results_b if r.get("is_duplicate_bug"))
    ok_b  = sum(1 for r in results_b if r.get("is_success"))

    print("\n" + "=" * 65)
    print("FINAL RESULT")
    print("=" * 65)
    print(f"Group A (no rule):   DUPLICATE_BUG={dup_a}/{N_PER_GROUP}  SUCCESS={ok_a}/{N_PER_GROUP}")
    print(f"Group B (with rule): DUPLICATE_BUG={dup_b}/{N_PER_GROUP}  SUCCESS={ok_b}/{N_PER_GROUP}")
    print()

    if dup_b < dup_a:
        reduction = dup_a - dup_b
        print(f"✓ PASS — conditional rule reduced duplicate-feature crashes by {reduction} "
              f"({dup_a}→{dup_b})")
    elif dup_b == dup_a == 0:
        print("~ INCONCLUSIVE — neither group triggered the bug (plan may not be specific enough)")
    elif dup_b == dup_a:
        print("~ INCONCLUSIVE — no difference between groups")
    else:
        print("✗ FAIL — conditional rule did not help (review rule wording)")
    print("=" * 65)


if __name__ == "__main__":
    main()
