#!/bin/bash

# Parallel batch runner for ML-Master with multiple seeds
# Usage: bash run_batch_parallel.sh <split_file> <n_seeds> [max_parallel]
# Example: bash run_batch_parallel.sh "$MLEBENCH_DIR/experiments/splits/dev.txt" 3 12
#
# Environment overrides (all have sensible, repo-relative defaults):
#   DATASET_DIR        Prepared MLE-bench data          (default: $HOME/.cache/mle-bench/data)
#   ML_MASTER_DIR      This ML-Master agent directory   (default: this script's directory)
#   MLEBENCH_DIR       mle-bench checkout                (default: <repo-root>/mle-bench)
#   CLEAN_WORKSPACES   Set to 1 to wipe previous run     (default: 0 — previous output is preserved)
#                      workspaces/logs before starting

set -e

# Resolve directories relative to this script so the runner is portable.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_MASTER_DIR="${ML_MASTER_DIR:-$SCRIPT_DIR}"
# Repo root is two levels up (…/Adversarial_EDA/ML-Master → repo root), mle-bench lives there.
MLEBENCH_DIR="${MLEBENCH_DIR:-$(cd "$SCRIPT_DIR/../../mle-bench" 2>/dev/null && pwd)}"

# Arguments
SPLIT_FILE=${1:-"${MLEBENCH_DIR}/experiments/splits/dev.txt"}
N_SEEDS=${2:-1}
MAX_PARALLEL=${3:-27}  # Maximum number of parallel jobs

# Configuration
DATASET_DIR="${DATASET_DIR:-$HOME/.cache/mle-bench/data}"
TOTAL_CPUS=240
CPUS_PER_TASK=5

# Output directory
BATCH_OUTPUT_DIR="./batch_results/$(basename ${SPLIT_FILE} .txt)_${N_SEEDS}seeds_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${BATCH_OUTPUT_DIR}

echo "=================================================="
echo "ML-Master Parallel Batch Runner with Multiple Seeds"
echo "=================================================="
echo "Split file: ${SPLIT_FILE}"
echo "Number of seeds: ${N_SEEDS}"
echo "Max parallel jobs: ${MAX_PARALLEL}"
echo "Total CPUs: ${TOTAL_CPUS}"
echo "CPUs per task: ${CPUS_PER_TASK}"
echo "Output directory: ${BATCH_OUTPUT_DIR}"
echo "=================================================="
echo ""

# Check if split file exists
if [ ! -f "${SPLIT_FILE}" ]; then
    echo "ERROR: Split file not found: ${SPLIT_FILE}"
    exit 1
fi

# Read competitions from split file
mapfile -t COMPETITIONS < <(grep -v '^#' "${SPLIT_FILE}" | grep -v '^$')
TOTAL_COMPS=${#COMPETITIONS[@]}

echo "Total competitions: ${TOTAL_COMPS}"
echo "Total runs: $((TOTAL_COMPS * N_SEEDS))"
echo ""

# Old workspaces/logs are preserved by default to avoid destroying prior results.
# Set CLEAN_WORKSPACES=1 to explicitly wipe them before this batch.
if [ "${CLEAN_WORKSPACES:-0}" = "1" ]; then
    echo "CLEAN_WORKSPACES=1 → removing previous ML-Master run workspaces/logs..."
    rm -rf "${ML_MASTER_DIR}/workspaces/run"/* "${ML_MASTER_DIR}/logs/run"/*
    echo "Cleanup complete."
else
    echo "Preserving existing workspaces/logs (set CLEAN_WORKSPACES=1 to wipe them)."
fi
echo ""

# Create a task list file
TASK_LIST="${BATCH_OUTPUT_DIR}/task_list.txt"
> ${TASK_LIST}  # Clear file

# Generate all tasks (competition x seed combinations)
for comp in "${COMPETITIONS[@]}"; do
    for seed in $(seq 1 ${N_SEEDS}); do
        echo "${comp} ${seed}" >> ${TASK_LIST}
    done
done

TOTAL_TASKS=$(wc -l < ${TASK_LIST})
echo "Generated ${TOTAL_TASKS} tasks"
echo ""

# Save configuration
cat > ${BATCH_OUTPUT_DIR}/batch_config.txt <<EOF
Split file: ${SPLIT_FILE}
Number of seeds: ${N_SEEDS}
Max parallel jobs: ${MAX_PARALLEL}
Total CPUs: ${TOTAL_CPUS}
CPUs per task: ${CPUS_PER_TASK}
Total competitions: ${TOTAL_COMPS}
Total tasks: ${TOTAL_TASKS}
Dataset directory: ${DATASET_DIR}
Started at: $(date)
EOF

# Function to run a single task
run_task() {
    local comp=$1
    local seed=$2
    local cpu_start=$3
    local cpu_end=$4
    local task_id="${comp}_seed${seed}"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${task_id} on CPUs ${cpu_start}-${cpu_end}"

    # Check if competition data exists
    if [ ! -d "${DATASET_DIR}/${comp}/prepared/public" ]; then
        echo "Preparing ${comp}..."
        if (cd "${MLEBENCH_DIR}" && mlebench prepare -c "${comp}") 2>&1 | tee ${BATCH_OUTPUT_DIR}/${task_id}_prepare.log; then
            echo "Successfully prepared ${comp}"
        else
            echo "ERROR: Failed to prepare ${comp}"
            echo "${task_id}: SKIPPED (data preparation failed)" >> ${BATCH_OUTPUT_DIR}/summary.txt
            return 1
        fi
    fi

    # Check if instructions exist
    DESC_FILE="${ML_MASTER_DIR}/dataset/full_instructions/${comp}/full_instructions.txt"
    if [ ! -f "${DESC_FILE}" ]; then
        echo "WARNING: Instructions not found: ${DESC_FILE}"
        echo "${task_id}: SKIPPED (instructions not found)" >> ${BATCH_OUTPUT_DIR}/summary.txt
        return 1
    fi

    # Set CPU affinity and run
    START_TIME=$(date +%s)

    if (cd "${ML_MASTER_DIR}" && taskset -c ${cpu_start}-${cpu_end} bash run.sh "${comp}" "${seed}") 2>&1 | tee ${BATCH_OUTPUT_DIR}/${task_id}_run.log; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed: ${task_id} in ${DURATION}s"
        echo "${task_id}: SUCCESS (${DURATION}s)" >> ${BATCH_OUTPUT_DIR}/summary.txt
        return 0
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        if [ ${EXIT_CODE} -eq 124 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⏱ TIMEOUT: ${task_id}"
            echo "${task_id}: TIMEOUT (${DURATION}s)" >> ${BATCH_OUTPUT_DIR}/summary.txt
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ FAILED: ${task_id} (exit code ${EXIT_CODE})"
            echo "${task_id}: FAILED (exit code ${EXIT_CODE}, ${DURATION}s)" >> ${BATCH_OUTPUT_DIR}/summary.txt
        fi
        return ${EXIT_CODE}
    fi
}

export -f run_task
export BATCH_OUTPUT_DIR
export DATASET_DIR
export ML_MASTER_DIR
export MLEBENCH_DIR

# Use GNU parallel if available, otherwise fall back to simple background jobs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for execution"
    echo ""

    # GNU parallel with CPU affinity
    cat ${TASK_LIST} | parallel -j ${MAX_PARALLEL} --colsep ' ' \
        'cpu_start=$((({%} - 1) * '"${CPUS_PER_TASK}"')); cpu_end=$(($cpu_start + '"${CPUS_PER_TASK}"' - 1)); run_task {1} {2} $cpu_start $cpu_end'

else
    echo "GNU parallel not found, using background jobs"
    echo "Note: Install GNU parallel for better performance: sudo apt-get install parallel"
    echo ""

    # Simple background job management
    declare -a PIDS
    job_count=0
    task_num=0

    while IFS=' ' read -r comp seed; do
        # Calculate CPU range for this job
        cpu_start=$((job_count * CPUS_PER_TASK))
        cpu_end=$((cpu_start + CPUS_PER_TASK - 1))

        # Run in background
        run_task "${comp}" "${seed}" ${cpu_start} ${cpu_end} &
        PIDS[$job_count]=$!

        ((job_count++))
        ((task_num++))

        # If we've reached max parallel jobs, wait for one to finish
        if [ ${job_count} -ge ${MAX_PARALLEL} ]; then
            echo "Waiting for a job to complete (${job_count}/${MAX_PARALLEL} running)..."
            wait -n  # Wait for any job to finish
            ((job_count--))
        fi

        echo "Progress: ${task_num}/${TOTAL_TASKS} tasks launched"
    done < ${TASK_LIST}

    # Wait for all remaining jobs
    echo "Waiting for remaining jobs to complete..."
    wait
fi

echo ""
echo "=================================================="
echo "All tasks completed!"
echo "=================================================="

# Generate summary
if [ -f ${BATCH_OUTPUT_DIR}/summary.txt ]; then
    SUCCESS_COUNT=$(grep -c "SUCCESS" ${BATCH_OUTPUT_DIR}/summary.txt 2>/dev/null || echo 0)
    FAIL_COUNT=$(grep -c "FAILED" ${BATCH_OUTPUT_DIR}/summary.txt 2>/dev/null || echo 0)
    TIMEOUT_COUNT=$(grep -c "TIMEOUT" ${BATCH_OUTPUT_DIR}/summary.txt 2>/dev/null || echo 0)
    SKIP_COUNT=$(grep -c "SKIPPED" ${BATCH_OUTPUT_DIR}/summary.txt 2>/dev/null || echo 0)

    echo "Successful: ${SUCCESS_COUNT}"
    echo "Failed: ${FAIL_COUNT}"
    echo "Timeout: ${TIMEOUT_COUNT}"
    echo "Skipped: ${SKIP_COUNT}"

    # Add to config
    cat >> ${BATCH_OUTPUT_DIR}/batch_config.txt <<EOF

Completed at: $(date)
Successful: ${SUCCESS_COUNT}
Failed: ${FAIL_COUNT}
Timeout: ${TIMEOUT_COUNT}
Skipped: ${SKIP_COUNT}
EOF
else
    echo "No summary file found"
fi

echo "Results saved to: ${BATCH_OUTPUT_DIR}"
echo "=================================================="
echo ""
echo "Summary of results:"
cat ${BATCH_OUTPUT_DIR}/summary.txt 2>/dev/null || echo "No results recorded"
