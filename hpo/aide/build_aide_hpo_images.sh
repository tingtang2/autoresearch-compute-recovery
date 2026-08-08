#!/bin/bash
# Build all AIDE HPO experiment images after installing agents into mle-bench/agents/.
# Run from the mle-bench directory (after following hpo/aide/README.md install steps).
set -euo pipefail

export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

for agent in aide aide_hpo_code aide_hpo_prompt aide_hpo_both; do
  echo "=== Building $agent ==="
  docker build --platform=linux/amd64 -t "$agent" "agents/$agent/" \
    --build-arg SUBMISSION_DIR="$SUBMISSION_DIR" \
    --build-arg LOGS_DIR="$LOGS_DIR" \
    --build-arg CODE_DIR="$CODE_DIR" \
    --build-arg AGENT_DIR="$AGENT_DIR"
done

echo "Done. Images: aide, aide_hpo_code, aide_hpo_prompt, aide_hpo_both"
