# Thompson Sampling in MLEvolve

MLEvolve runs an MCTS-style tree search over candidate solutions. At the **root
level** (i.e. when choosing which *draft branch* to expand next) it can select a
branch in one of two ways:

- **UCT (default MCTS)** — pick the branch with the highest Upper Confidence Bound.
- **Thompson Sampling (TS)** — treat each branch root as a Bernoulli bandit arm with
  a `Beta(alpha, beta)` posterior, draw one sample per branch, and expand the branch
  with the highest sample.

This document explains **how to switch TS on/off** and **which files implement it**.

All relative paths and commands below assume you have completed the environment
setup in [`README.md`](README.md) and are working from
`ThompsonSampling/MLEvolve/`:

```bash
cd ThompsonSampling/MLEvolve
```

---

## 1. How to switch it on

Thompson Sampling is controlled entirely from `config/config.yaml`, under
`agent.search`:

```yaml
agent:
  search:
    # ---- Thompson Sampling for branch selection ----
    # When true, replaces UCT at the root level with Beta(alpha, beta) sampling.
    # Branch roots with better descendants accumulate higher alpha over time.
    use_thompson_sampling: true      # <-- master switch (true = TS, false = UCT)
    thompson_prior_alpha: 1.0        # Beta(1,1) = uniform prior (no bias)
    thompson_prior_beta: 1.0
```

- Set **`use_thompson_sampling: true`** to use Thompson Sampling.
- Set **`use_thompson_sampling: false`** (the *baseline*) to fall back to UCT.

You can also override it on the command line without editing the file (config is
merged with CLI args via OmegaConf in `config/__init__.py`):

```bash
python run.py \
    data_dir=... desc_file=... \
    agent.search.use_thompson_sampling=true
```

> **Note on the priors.** `thompson_prior_alpha` / `thompson_prior_beta` are declared
> in the config schema for tuning, but the running engine currently initializes every
> `SearchNode` with the class defaults `alpha = 1.0`, `beta = 1.0`
> (`engine/search_node.py`), which already equal the uniform `Beta(1,1)` prior. Only
> `use_thompson_sampling` changes behaviour today; changing the prior values has no
> effect unless the node initializer is wired to read them.

---

## 2. What happens when it's on

Two things change once `use_thompson_sampling: true`:

1. **Branch selection** — at the root, instead of the UCT `argmax`, MLEvolve draws
   `random.betavariate(alpha, beta)` for each unlocked branch and expands the branch
   with the largest draw.
2. **Posterior updates** — after every node is evaluated, the branch root's
   `Beta(alpha, beta)` is updated with a normalized reward in `[0, 1]`
   (`alpha += reward`, `beta += 1 - reward`). Better-performing branches accumulate
   higher `alpha` and get sampled more often over time.

When it's **off**, both of these are skipped and the search uses UCT with the
piecewise-decayed exploration constant.

---

## 3. Files involved

The active entry point is **`run.py`**, which imports `AgentSearch` from
`engine/agent_search.py` and loads config via `config/__init__.py`. The relevant
Thompson Sampling code lives in the following files:

| File | Role in Thompson Sampling |
| --- | --- |
| `config/config.yaml` | Defines the switch and priors: `agent.search.use_thompson_sampling`, `thompson_prior_alpha`, `thompson_prior_beta`. |
| `config/__init__.py` | Active config loader/validator. `SearchConfig` declares `use_thompson_sampling`, `thompson_prior_alpha`, `thompson_prior_beta`; merges YAML + CLI overrides. |
| `engine/node_selection.py` | **Branch selection.** `_thompson_select()` samples `Beta(alpha, beta)` per branch and returns the argmax sample. In `select()`, the root chooses `_thompson_select(...)` when the flag is set, otherwise UCT. |
| `engine/evaluation.py` | **Posterior updates.** `get_normalized_reward()` maps a node's result to `[0, 1]`; `_get_branch_root()` finds the owning branch root; `_update_thompson_branch()` applies the Beta update (no-op when the flag is off). |
| `engine/search_node.py` | **State.** `SearchNode.alpha` / `SearchNode.beta` fields, `update_thompson(reward)` (the Beta update), and `p_mean()`. |
| `engine/agent_search.py` | **Orchestration.** Logs `"[MCTS] Thompson Sampling: enabled/disabled"` at startup and calls `evaluation._update_thompson_branch(self, result_node)` after each successful node evaluation. |
| `run.py` | Entry point; constructs `AgentSearch` and drives the search loop. |

> Sibling files `engine/new_agent_search.py`, `engine/old_agent_search.py`,
> `new_run.py`, and `config/new_init.py` are alternate/variant copies that contain the
> same TS hooks but are **not** used by the default `run.py` entry point.

---

## 4. Quick verification

At the start of a run, the log line tells you which mode is active:

```
[MCTS] Thompson Sampling: enabled        # use_thompson_sampling: true
[MCTS] Thompson Sampling: disabled (UCT) # use_thompson_sampling: false
```

During the run you'll also see per-selection and per-update lines, e.g.:

```
[Thompson] Selected branch node 1a2b3c4d (sample=0.812, μ=0.640, n=7)
[Thompson] Branch root 1a2b3c4d updated with reward=1.000 (α=6.20, β=2.80)
```

---

## 5. Baseline vs. TS experiments

To compare a batch of **baseline** runs (`use_thompson_sampling: false`) against a
batch of **TS** runs (`use_thompson_sampling: true`), point each batch's grading
reports at `analysis/analyze_runs.py`, which produces a per-method CSV for each and a
LaTeX comparison table. See `analysis/analyze_runs.py --help`.
