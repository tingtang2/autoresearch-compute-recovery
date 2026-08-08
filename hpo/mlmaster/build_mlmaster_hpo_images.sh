#!/bin/bash
# Build all ML-Master HPO experiment images after installing agents into mle-bench/agents/.
# Run from the mle-bench directory (after following hpo/mlmaster/README.md install steps).
set -euo pipefail

export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

if ! docker image inspect mlebench-env >/dev/null 2>&1; then
  echo "WARNING: mlebench-env image not found. Build it first if these builds fail."
fi

for agent in mlmaster mlmaster_hpo_code mlmaster_hpo_prompt mlmaster_hpo_both; do
  echo "=== Building $agent ==="
  docker build --platform=linux/amd64 -t "$agent" "agents/$agent/" \
    --build-arg SUBMISSION_DIR="$SUBMISSION_DIR" \
    --build-arg LOGS_DIR="$LOGS_DIR" \
    --build-arg CODE_DIR="$CODE_DIR" \
    --build-arg AGENT_DIR="$AGENT_DIR"
done

echo "Done. Images: mlmaster, mlmaster_hpo_code, mlmaster_hpo_prompt, mlmaster_hpo_both"
