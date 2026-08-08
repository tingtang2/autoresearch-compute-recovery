# Thompson Sampling on AIDE

This document explains how to **run** AIDE with a **Thompson Sampling** (MCTS-style) node-selection
policy over the solution tree, how to tune it, and how to run the matching **control** (Thompson
Sampling disabled). It is a runnable treatment/control pair, not just a settings reference.

AIDE explores a tree of candidate solutions: each node is a program, and children are produced by
**drafting** a new approach, **improving** a working node, or **debugging** a buggy node. When the
search **backtracks** (e.g. after repeated errors) it must pick which sibling/branch to expand next.
Thompson Sampling replaces the default random pick with a Beta-posterior bandit that balances
**exploration** (under-visited branches) against **exploitation** (branches that score well).

---

## Important: which entrypoint actually runs MCTS

The MCTS / Thompson-Sampling logic lives in the **backtracking** agent, not the stock one:

| Component | File (under `aideml/aide/`) |
|---|---|
| Backtracking agent (reads `use_mcts`) | `agent_backtrack.py` |
| Selectors (Thompson / UCB1 / random) | `mcts_backtrack.py` |
| Config schema + defaults | `utils/config_backtrack.py`, `utils/config_backtrack.yaml` |
| **MCTS entrypoint** | `run_backtrack.py` |

The stock `aide` command (`aide.run:run`) uses `aide/agent.py`, which has **no** `use_mcts` support —
so passing `agent.search.use_mcts=true` to plain `aide` does nothing. **You must use the
`aide.run_backtrack` entrypoint** (added in this repo), which loads `config_backtrack.yaml` and uses
`agent_backtrack.Agent`.

This runs the **vendored AIDE at the repo root** (`llms-for-mle-bench/aideml/`); it does **not** rely
on the `mle-bench` submodule (pinned to upstream), which is left untouched.

---

## TL;DR

- **Enable Thompson Sampling:** run `python -m aide.run_backtrack ... agent.search.use_mcts=true agent.search.mcts_method=thompson`.
- **Control (disable it):** same command with `agent.search.use_mcts=false` (falls back to random selection).
- **Turn on early backtracking on repeated errors:** set `agent.search.error_backtrack_threshold` (e.g. `3`).

---

## Setup (once)

```bash
cd llms-for-mle-bench
python -m venv .venv && source .venv/bin/activate
pip install -e aideml
export OPENAI_API_KEY=sk-...
```

You also need a prepared competition and a task-description file:

- `data_dir` → `.../<competition>/prepared/public`
- `desc_file` → a plaintext task description (e.g. `.../full_instructions.txt` or the competition `description.md`)

Use `mlebench prepare -c <competition>` from the `mle-bench` checkout to produce the data.

---

## Run it: treatment vs. control

Run both with the **same** competition, `exp_name` aside, and the same step budget — the only
difference is `use_mcts`.

**Treatment — Thompson Sampling ON:**

```bash
python -m aide.run_backtrack \
  data_dir=/path/to/<competition>/prepared/public \
  desc_file=/path/to/<competition>/full_instructions.txt \
  exp_name=aide_mcts \
  agent.search.use_mcts=true \
  agent.search.mcts_method=thompson \
  agent.search.error_backtrack_threshold=3
```

**Control — Thompson Sampling OFF (random node selection):**

```bash
python -m aide.run_backtrack \
  data_dir=/path/to/<competition>/prepared/public \
  desc_file=/path/to/<competition>/full_instructions.txt \
  exp_name=aide_control \
  agent.search.use_mcts=false \
  agent.search.error_backtrack_threshold=3
```

Any `agent.search.*` value can be overridden on the CLI (OmegaConf merges CLI args over
`config_backtrack.yaml`). The example values you may have seen elsewhere map directly onto real,
honored settings:

```bash
python -m aide.run_backtrack \
  data_dir=... desc_file=... exp_name=aide_mcts \
  agent.search.use_mcts=true \
  agent.search.max_debug_depth=5 \
  agent.search.debug_prob=0.7 \
  agent.search.error_backtrack_threshold=3
```

---

## Configuration reference (`agent.search`)

Defaults live in `aideml/aide/utils/config_backtrack.yaml`; override any on the CLI.

| Key | Default | Description |
| --- | --- | --- |
| `use_mcts` | `true` | Master switch. `true` → posterior-sampling node selection at backtrack points; `false` → random selection (control). |
| `mcts_method` | `thompson` | Selection algorithm: `thompson` (recommended), `ucb1`, or `random`. |
| `error_backtrack_threshold` | `5` | Backtrack after seeing the **same** error this many times (early escape from a stuck branch). Set lower (e.g. `3`) to backtrack sooner. |
| `max_debug_depth` | `3` | Max consecutive debug steps down a single branch before selection must move elsewhere. |
| `debug_prob` | `0.5` | Probability of debugging a fixable bug (vs. exploring elsewhere). |
| `num_drafts` | `5` | Number of independent root drafts. |
| `thompson_prior_alpha` | `1.0` | Beta prior "successes" (higher = more optimistic). Used when `mcts_method=thompson`. |
| `thompson_prior_beta` | `1.0` | Beta prior "failures" (higher = more pessimistic). Used when `mcts_method=thompson`. |
| `ucb_exploration_weight` | `1.414` | Exploration constant, used when `mcts_method=ucb1`. |

---

## How Thompson Sampling works here

At a backtrack point, `agent_backtrack.py` asks the selector (`mcts_backtrack.py`) to choose among
candidate nodes:

1. Each node keeps a `Beta(alpha, beta)` posterior over its "reward" (normalized validation metric;
   buggy nodes score 0).
2. The selector draws **one sample per node** and picks the node with the highest sample (so
   under-explored nodes with wide posteriors still get chances).
3. After a node executes, its reward updates the posterior (`update_reward`), informing later picks.

`error_backtrack_threshold` decides *when* to backtrack (after N identical errors); `use_mcts` /
`mcts_method` decide *which* branch to jump to; `max_debug_depth` and `debug_prob` govern how much
debugging happens before moving on.

---

## Verifying it's active

With `use_mcts=true` you should see log lines like (in `logs/<exp_name>/aide.log`):

```
[search policy] use_mcts=True method=thompson error_backtrack_threshold=3
[MCTS ENABLED] Using thompson for node selection
[search policy] Same error seen 3 times, triggering backtrack
[Thompson Sampling] Selected node abc123 (sample=0.78, posterior: μ=0.75, σ=0.12, n=3.0)
```

With `use_mcts=false` you'll instead see `[MCTS DISABLED] Using random node selection`.

You can also exercise the selector directly (no data or API key needed):

```bash
python -c "
from aide.mcts_backtrack import create_selector
s = create_selector('thompson')
print('selector:', type(s).__name__)
"
```

---

## Combining with adversarial EDA

To inject an adversarial EDA finding *and* use Thompson Sampling, use the EDA launcher with
`--agent backtrack` (see `Adversarial_EDA/AIDE/README.md`):

```bash
EDA_FINDINGS="..." python Adversarial_EDA/AIDE/run_with_eda.py --agent backtrack -- \
  data_dir=... desc_file=... exp_name=eda_mcts agent.search.use_mcts=true
```

---

## Notes & caveats

- The **control** run is genuinely clean: no EDA text is injected unless you opt in via the EDA hook
  (`EDA_FINDINGS`). Both previous injection routes are neutralized — the module-level `EDA_MEMORY` is
  empty in `aide.agent` and `aide.agent_backtrack`, **and** neither agent's `_draft` hard-codes an EDA
  block (the former `"Prior Analysis"` text was removed to match `aide/agent.py`). Both routes are
  guarded by `Adversarial_EDA/AIDE/test_eda_hook.py`; run it after `pip install -e aideml`, and keep
  this clean-control claim only while it passes:

  ```bash
  python Adversarial_EDA/AIDE/test_eda_hook.py
  # -> "All eda_hook regression checks passed."
  ```

- `debug_prob` is a probability in `[0, 1]`; `error_backtrack_threshold` and `max_debug_depth` are
  positive integers (`0`/unset disables early error backtracking).
- This is the **standalone** run path (`python -m aide.run_backtrack`), not the containerized
  `mlebench run-agent` harness. Wiring the Docker harness would additionally require registering a
  backtrack agent variant inside `mle-bench/agents/` (which we intentionally do not modify, to keep
  the submodule synced to upstream).
- These commands are verified for syntax/wiring but were **not** executed end-to-end here (a real run
  needs prepared data, compute, and an API key).
