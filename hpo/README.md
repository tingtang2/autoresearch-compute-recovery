# HPO (Hyperparameter Optimization) interventions

Reproducible 2×2 factorial for **prompt HPO** vs **control-loop HPO** on MLE-bench agents.

| Condition | Prompt directive (`additional_notes.txt`) | Control-loop scoring |
|-----------|-------------------------------------------|----------------------|
| baseline | no | no |
| hpo-code | no | yes |
| hpo-prompt | yes (`HYPERPARAMETER OPTIMIZATION DIRECTIVE`) | no |
| hpo-both | yes | yes |

## Layout

```
hpo/
  README.md
  competitions_9.txt           the 9-task set

  aide/                        AIDE variants + install/run docs
    README.md
    build_aide_hpo_images.sh
    agents/
      aide/                    baseline
      aide_hpo_code/           control-loop only
      aide_hpo_prompt/         prompt directive only
      aide_hpo_both/           both

  mlmaster/                    ML-Master variants + install/run docs
    README.md
    build_mlmaster_hpo_images.sh
    agents/
      mlmaster/                baseline
      mlmaster_hpo_code/       control-loop only
      mlmaster_hpo_prompt/     prompt directive only
      mlmaster_hpo_both/       both
```

`aide/agents/*` and `mlmaster/agents/*` are self-contained MLE-bench agent trees
(Dockerfile, `config.yaml`, `start.sh`, `additional_notes.txt`, and agent source).
They install into the `mle-bench` submodule’s `agents/` directory — this folder is
**not** a copy of the harness.

## Install (AIDE)

```bash
cd <repo root>          # the dir containing mle-bench/ and hpo/

cp -r hpo/aide/agents/aide/.            mle-bench/agents/aide/
cp -r hpo/aide/agents/aide_hpo_code/.   mle-bench/agents/aide_hpo_code/
cp -r hpo/aide/agents/aide_hpo_prompt/. mle-bench/agents/aide_hpo_prompt/
cp -r hpo/aide/agents/aide_hpo_both/.   mle-bench/agents/aide_hpo_both/
```

## Install (ML-Master)

```bash
cd <repo root>

cp -r hpo/mlmaster/agents/mlmaster/.            mle-bench/agents/mlmaster/
cp -r hpo/mlmaster/agents/mlmaster_hpo_code/.   mle-bench/agents/mlmaster_hpo_code/
cp -r hpo/mlmaster/agents/mlmaster_hpo_prompt/. mle-bench/agents/mlmaster_hpo_prompt/
cp -r hpo/mlmaster/agents/mlmaster_hpo_both/.   mle-bench/agents/mlmaster_hpo_both/
```

Check registry:

```bash
cd mle-bench
python -c "from agents.registry import registry; print([a.id for a in registry.get_all_agents() if 'hpo' in a.id or a.id in ('aide','mlmaster')])"
```

## Build + run

See [`aide/README.md`](aide/README.md) and [`mlmaster/README.md`](mlmaster/README.md).

## What each axis does

- **Prompt HPO** — appends a `HYPERPARAMETER OPTIMIZATION DIRECTIVE` into the task
  instructions via `start.sh` (`envsubst` on `additional_notes.txt`).
- **Control-loop HPO** —
  - **AIDE:** after each execution, scores whether the generated code does real
    hyperparameter tuning (`_score_hyperparameter_tuning` in `aideml/aide/agent.py`)
    and uses that signal in search / metric shaping.
  - **ML-Master:** same idea in `agent/mcts_agent.py` (scoring / reward shaping /
    baseline-phase policy), not prompt text alone.
