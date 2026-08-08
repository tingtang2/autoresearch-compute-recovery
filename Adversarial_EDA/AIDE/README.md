# Adversarial EDA injection for AIDE

This directory makes AIDE runnable with an **injected, adversarial "EDA finding"** so
you can measure how a (possibly misleading) exploratory-data-analysis result changes
the agent's solutions — versus a clean **baseline** with no injection.

It runs the **vendored AIDE at the repo root** (`llms-for-mle-bench/aideml/`), so it
does **not** depend on the `mle-bench` submodule (which is pinned to upstream). The
submodule is untouched.

## How it works

AIDE's agent modules define a module-level `EDA_MEMORY` string that is prepended to the
"Memory" section of the draft/improve prompts:

```python
prompt["Memory"] = EDA_MEMORY + self.journal.generate_summary()
```

`eda_hook.py` sets `EDA_MEMORY` at runtime from the `EDA_FINDINGS` environment variable
(wrapping it in a `Design / Results / Validation Metric` block). It patches both
`aide.agent` (baseline) and `aide.agent_backtrack` (MCTS) if present. If `EDA_FINDINGS`
is unset/empty the hook is a **no-op**, giving clean baseline behaviour. `run_with_eda.py`
applies the hook and then forwards all `key=value` args to the chosen AIDE entrypoint.

The raw findings text is kept in **`Adversarial_EDA/eda_findings.txt`** — the single
source of truth shared with the LLM judge (`analyze_eda_impact.py`) and the ML-Master
patcher (`implement_EDA.py`), so every agent injects and is scored against identical
text. Set `EDA_FINDINGS` from that file rather than pasting an abbreviated string.

## Setup (once)

```bash
cd llms-for-mle-bench
python -m venv .venv && source .venv/bin/activate
pip install -e aideml
export OPENAI_API_KEY=sk-...
```

You also need a prepared competition (`data_dir` points at `.../prepared/public`) and a
task-description file (`desc_file`). Use `mlebench prepare -c <competition>` from the
`mle-bench` checkout to produce the data.

## Baseline vs. injected (the experiment pair)

Run both with the **same** competition, seed, and step budget; the only difference is
whether `EDA_FINDINGS` is set.

**Control (baseline — no EDA injection):**

```bash
# EDA_FINDINGS explicitly absent -> hook is a no-op
env -u EDA_FINDINGS python Adversarial_EDA/AIDE/run_with_eda.py -- \
  data_dir=/path/to/<competition>/prepared/public \
  desc_file=/path/to/<competition>/full_instructions.txt \
  exp_name=eda_baseline
```

**Treatment (injected EDA finding):**

```bash
export EDA_FINDINGS="$(cat Adversarial_EDA/eda_findings.txt)"

python Adversarial_EDA/AIDE/run_with_eda.py -- \
  data_dir=/path/to/<competition>/prepared/public \
  desc_file=/path/to/<competition>/full_instructions.txt \
  exp_name=eda_injected
```

**Injected EDA + Thompson Sampling** (combine with the MCTS entrypoint — see
`ThompsonSampling/AIDE/README_AIDE.MD`):

```bash
export EDA_FINDINGS="$(cat Adversarial_EDA/eda_findings.txt)"
python Adversarial_EDA/AIDE/run_with_eda.py --agent backtrack -- \
  data_dir=/path/to/<competition>/prepared/public \
  desc_file=/path/to/<competition>/full_instructions.txt \
  exp_name=eda_mcts \
  agent.search.use_mcts=true
```

## Verifying the injection took effect

The hook logs to the `aide` logger, e.g.:

```
[eda_hook] Injected EDA memory into aide.agent.EDA_MEMORY (312 chars).
```

You can also grep the saved run: the injected block appears in the agent's Memory
context in `logs/<exp_name>/aide.verbose.log`.

## Files

| File | Purpose |
|---|---|
| `eda_hook.py` | Sets `EDA_MEMORY` from `EDA_FINDINGS`; no-op if unset. Patches baseline + backtrack agents. |
| `run_with_eda.py` | Applies the hook, then launches `aide.run` (baseline) or `aide.run_backtrack` (MCTS). |
| `test_eda_hook.py` | Regression test: both agent modules default to an empty `EDA_MEMORY`, neither `_draft` hard-codes EDA text, and injection happens only when `EDA_FINDINGS` is set. |

## Regression test

A run with `EDA_FINDINGS` unset must be truly clean (no baked-in adversarial text). Two routes
previously violated this; both are now guarded:

1. **`EDA_MEMORY`** — both `aide.agent` and `aide.agent_backtrack` ship an **empty** module-level
   `EDA_MEMORY`.
2. **`_draft` prompt** — neither agent's `_draft` hard-codes an EDA block. `aide.agent_backtrack`
   previously embedded a `"Prior Analysis"` finding that fired on every backtracking run regardless
   of `EDA_FINDINGS`/`use_mcts`; it has been removed to match `aide/agent.py`.

Verify with:

```bash
pip install -e aideml            # from repo root, once
python Adversarial_EDA/AIDE/test_eda_hook.py
# -> "All eda_hook regression checks passed."
```

## Analysis

To score the impact of injected findings across baseline/treatment runs, see
`Adversarial_EDA/llm-as-a-judge/analyze_eda_impact.py`.

> **Note:** these commands have been verified for syntax and wiring, but not executed
> end-to-end here (a real run needs prepared competition data, compute, and an API key).
