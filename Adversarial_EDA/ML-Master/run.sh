#!/bin/bash
set -x # Print commands and their arguments as they are executed

AGENT_DIR=./
EXP_ID=${1:-spaceship-titanic}  # Accept competition ID as first argument
SEED=${2:-42}                    # Accept seed as second argument
# Prepared MLE-bench data. Override by exporting DATASET_DIR before running.
dataset_dir="${DATASET_DIR:-$HOME/.cache/mle-bench/data}"
MEMORY_INDEX=0

code_model=gpt-5-mini-2025-08-07
code_temp=1.0
code_base_url="https://api.openai.com/v1"
code_api_key="${OPENAI_API_KEY}"  # set via env var; do NOT commit real keys

feedback_model=gpt-5-mini-2025-08-07
feedback_temp=1.0
feedback_base_url="https://api.openai.com/v1"
feedback_api_key="${OPENAI_API_KEY}"  # set via env var; do NOT commit real keys

start_cpu=0
CPUS_PER_TASK=22  # Changed from 22 to allow 12 parallel jobs (240/12=20)
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

# Wall-clock budget per run, in seconds. Default is 2 hours; override via env,
# e.g. TIME_LIMIT_SECS=300 for a quick smoke test, or 43200 for a 12h run.
TIME_LIMIT_SECS="${TIME_LIMIT_SECS:-7200}"

cd ${AGENT_DIR}
export MEMORY_INDEX
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export STEP_LIMIT=500

mkdir -p ${AGENT_DIR}/logs

# use the mirror if needed
# export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} timeout $TIME_LIMIT_SECS python main_mcts.py \
  dataset_dir="${dataset_dir}" \
  data_dir="${dataset_dir}/${EXP_ID}/prepared/public" \
  desc_file="./dataset/full_instructions/${EXP_ID}/full_instructions.txt" \
  exp_name="${EXP_ID}_seed${SEED}_mcts_comp_validcheck_[cpu-${start_cpu}-${end_cpu}]" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  agent.code.model=$code_model \
  agent.code.temp=$code_temp \
  agent.code.base_url=$code_base_url \
  agent.code.api_key=$code_api_key \
  agent.feedback.model=$feedback_model \
  agent.feedback.temp=$feedback_temp \
  agent.feedback.base_url=$feedback_base_url \
  agent.feedback.api_key=$feedback_api_key \
  agent.steerable_reasoning=false

if [ $? -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi
