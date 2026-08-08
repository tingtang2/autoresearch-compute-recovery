# AIDE HPO variants

Four self-contained MLE-bench agent trees for the HPO factorial (baseline / code / prompt / both).

## Conditions

| Agent id | Folder under `agents/` | Docker image tag | `agent.py` HPO scoring | HPO directive in notes |
|----------|------------------------|------------------|------------------------|-------------------------|
| `aide` | `aide` | `aide` | no | no |
| `aide-hpo-code` | `aide_hpo_code` | `aide_hpo_code` | yes | no |
| `aide-hpo-prompt` | `aide_hpo_prompt` | `aide_hpo_prompt` | no | yes |
| `aide-hpo-both` | `aide_hpo_both` | `aide_hpo_both` | yes | yes |

Differing files (everything else is shared packaging):

- Control-loop: `aideml/aide/agent.py` (`_score_hyperparameter_tuning`, reward shaping)
- Prompt: `additional_notes.txt` (`HYPERPARAMETER OPTIMIZATION DIRECTIVE` block)

## Install

From the **llms-for-mle-bench** repo root (sibling of `mle-bench/`):

```bash
cp -r hpo/aide/agents/aide/.            mle-bench/agents/aide/
cp -r hpo/aide/agents/aide_hpo_code/.   mle-bench/agents/aide_hpo_code/
cp -r hpo/aide/agents/aide_hpo_prompt/. mle-bench/agents/aide_hpo_prompt/
cp -r hpo/aide/agents/aide_hpo_both/.   mle-bench/agents/aide_hpo_both/
```

These Dockerfiles build **from the agent folder** (self-contained `aideml/` inside each tree),
unlike some other experiment folders that use a parent build context.

## Build images

```bash
cd mle-bench
export OPENAI_API_KEY=...   # do not commit keys
bash ../hpo/aide/build_aide_hpo_images.sh
```

Or manually:

```bash
cd mle-bench
export SUBMISSION_DIR=/home/submission LOGS_DIR=/home/logs CODE_DIR=/home/code AGENT_DIR=/home/agent
for agent in aide aide_hpo_code aide_hpo_prompt aide_hpo_both; do
  docker build --platform=linux/amd64 -t "$agent" "agents/$agent/" \
    --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
    --build-arg LOGS_DIR=$LOGS_DIR \
    --build-arg CODE_DIR=$CODE_DIR \
    --build-arg AGENT_DIR=$AGENT_DIR
done
```

## Run

Default time limit in each `config.yaml` is **7200s (2 hours)**.

```bash
cd mle-bench
export OPENAI_API_KEY=...

# Example: all four conditions on the 9-comp set (1 seed) — adjust workers to GPU count
python run_agent.py --agent-id aide \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9

python run_agent.py --agent-id aide-hpo-code \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9

python run_agent.py --agent-id aide-hpo-prompt \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9

python run_agent.py --agent-id aide-hpo-both \
  --competition-set ../hpo/competitions_9.txt --n-seeds 1 --n-workers 9
```

### Sanity checks after a run

- **Prompt on:** `aide.verbose.log` / prompts contain `HYPERPARAMETER OPTIMIZATION DIRECTIVE`
- **Code on:** logs contain `Scoring hyperparameter tuning for node ...`
- Baseline / prompt-only should **not** emit the scoring lines; baseline / code-only should
  **not** contain the directive in notes.

## Models

Configs use `gpt-5-mini-2025-08-07` for code and feedback (see each `config.yaml`). Change
there if you need a different model.
