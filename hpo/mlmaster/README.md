# ML-Master HPO variants

Four self-contained MLE-bench agent trees for the HPO factorial (baseline / code / prompt / both).

## Conditions

| Agent id | Folder under `agents/` | Docker image tag | `mcts_agent.py` HPO scoring | HPO directive in notes |
|----------|------------------------|------------------|-----------------------------|-------------------------|
| `mlmaster` | `mlmaster` | `mlmaster` | no | no |
| `mlmaster-hpo-code` | `mlmaster_hpo_code` | `mlmaster_hpo_code` | yes | no |
| `mlmaster-hpo-prompt` | `mlmaster_hpo_prompt` | `mlmaster_hpo_prompt` | no | yes |
| `mlmaster-hpo-both` | `mlmaster_hpo_both` | `mlmaster_hpo_both` | yes | yes |

Differing files (everything else is shared packaging):

- Control-loop: `agent/mcts_agent.py` (`_score_hyperparameter_tuning`, reward shaping, baseline-phase policy)
- Prompt: `additional_notes.txt` (`HYPERPARAMETER OPTIMIZATION DIRECTIVE` block)
- Config: `agent.require_hyperparameter_tuning` is `true` only for code / both

## Install

From the **llms-for-mle-bench** repo root (sibling of `mle-bench/`):

```bash
cp -r hpo/mlmaster/agents/mlmaster/.            mle-bench/agents/mlmaster/
cp -r hpo/mlmaster/agents/mlmaster_hpo_code/.   mle-bench/agents/mlmaster_hpo_code/
cp -r hpo/mlmaster/agents/mlmaster_hpo_prompt/. mle-bench/agents/mlmaster_hpo_prompt/
cp -r hpo/mlmaster/agents/mlmaster_hpo_both/.   mle-bench/agents/mlmaster_hpo_both/
```

These Dockerfiles build **from the agent folder** (self-contained ML-Master source inside each tree).
They expect a pre-built `mlebench-env` base image (same convention as the main ML-Master agent).

## Build images

```bash
cd mle-bench
export OPENAI_API_KEY=...   # do not commit keys
bash ../hpo/mlmaster/build_mlmaster_hpo_images.sh
```

Or manually:

```bash
cd mle-bench
export SUBMISSION_DIR=/home/submission LOGS_DIR=/home/logs CODE_DIR=/home/code AGENT_DIR=/home/agent
for agent in mlmaster mlmaster_hpo_code mlmaster_hpo_prompt mlmaster_hpo_both; do
  docker build --platform=linux/amd64 -t "$agent" "agents/$agent/" \
    --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
    --build-arg LOGS_DIR=$LOGS_DIR \
    --build-arg CODE_DIR=$CODE_DIR \
    --build-arg AGENT_DIR=$AGENT_DIR
done
```

If a full rebuild fails on optional deps (e.g. `torch_cluster`), use an existing
`mlmaster:latest` and overlay only the differing files (`additional_notes.txt`,
`agent/mcts_agent.py`, configs) — the factorial only depends on those toggles.

## Run

Default time limit in each `config.yaml` is **7200s (2 hours)**. Smoke ids
(`mlmaster/smoke`, `mlmaster-hpo-code/smoke`, …) use 600s / 15 steps.

```bash
cd mle-bench
export OPENAI_API_KEY=...

# Example: all four conditions on the 9-comp set (1 seed) — adjust workers to GPU count
python run_agent.py --agent-id mlmaster \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9

python run_agent.py --agent-id mlmaster-hpo-code \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9

python run_agent.py --agent-id mlmaster-hpo-prompt \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9

python run_agent.py --agent-id mlmaster-hpo-both \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9
```

### Sanity checks after a run

- **Prompt on:** prompts / full instructions contain `HYPERPARAMETER OPTIMIZATION DIRECTIVE`
- **Code on:** logs contain scoring / HPO control-loop messages from `mcts_agent.py`
  (e.g. hyperparameter tuning score lines)
- Baseline / prompt-only should **not** use the scoring path; baseline / code-only should
  **not** contain the directive in notes

## Models

Configs use `gpt-5-mini` for code and feedback (see each `config.yaml`). Change
there if you need a different model.
