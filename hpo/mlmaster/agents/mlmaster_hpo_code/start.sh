#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
  MEMORY_INDEX=0
else
  HARDWARE="a CPU"
  MEMORY_INDEX=0
fi
export HARDWARE
export MEMORY_INDEX

# check that we can use the GPU in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')" || true
# check that we can use the GPU in TensorFlow
python -c "import tensorflow as tf; print('GPUs Available: ', tf.config.list_physical_devices('GPU'))" || true

# convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)

# Create full instructions file by combining MLE-bench instructions with additional notes
# Start with the base instructions from MLE-bench
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt

# Append additional notes (with environment variable substitution)
# Add a linebreak before the additional notes section
echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt

# Set up environment variables from config
EXP_ID=${COMPETITION_ID}
dataset_dir=/home  # MLE-bench mounts data at /home/data
data_dir=/home/data  # Public data is at /home/data
desc_file=${AGENT_DIR}/full_instructions.txt  # Use combined instructions file

# Create logs directory
mkdir -p ${AGENT_DIR}/logs

# Launch grading server in background (required by ML-Master)
# The grading server validates submissions
python -u grading_server.py \
  dataset_dir="${dataset_dir}" \
  data_dir="none" \
  desc_file="none" > ${LOGS_DIR}/grading_server.log 2>&1 &
GRADING_SERVER_PID=$!

# Wait a bit for server to start
sleep 2

# Cleanup function to kill grading server
cleanup() {
    if [ ! -z "$GRADING_SERVER_PID" ]; then
        kill $GRADING_SERVER_PID 2>/dev/null || true
    fi
}

# Function to copy submission file (call on exit to ensure it always runs)
copy_submission() {
    # Copy submission to expected location if it exists
    mkdir -p ${SUBMISSION_DIR}
    
    # Log to both stderr and a file for debugging
    DEBUG_LOG="/home/logs/copy_submission_debug.log"
    log_msg() {
        echo "$1" >&2
        echo "$1" >> "${DEBUG_LOG}" 2>/dev/null || true
    }
    
    log_msg "=== Starting copy_submission function ==="
    log_msg "SUBMISSION_DIR: ${SUBMISSION_DIR}"
    log_msg "AGENT_DIR: ${AGENT_DIR}"
    log_msg "Attempting to copy submission file from workspace..."
    
    if [ -d "${AGENT_DIR}/workspaces" ]; then
        log_msg "Found workspaces directory: ${AGENT_DIR}/workspaces"
        # First try to find best_submission/submission.csv (where ML-Master saves the best solution)
        BEST_SUBMISSION=$(find ${AGENT_DIR}/workspaces -type f -path "*/best_submission/submission.csv" 2>/dev/null | sort -r | head -1)
        if [ ! -z "$BEST_SUBMISSION" ]; then
            log_msg "Found best submission at: $BEST_SUBMISSION"
            if cp "$BEST_SUBMISSION" ${SUBMISSION_DIR}/submission.csv 2>>"${DEBUG_LOG}"; then
                log_msg "Successfully copied best submission to ${SUBMISSION_DIR}/submission.csv"
                return 0
            else
                log_msg "ERROR: Failed to copy best submission file"
            fi
        fi
        
        # Fallback to submission/submission.csv (intermediate location during execution)
        SUBMISSION_FILE=$(find ${AGENT_DIR}/workspaces -type f -path "*/submission/submission.csv" 2>/dev/null | sort -r | head -1)
        if [ ! -z "$SUBMISSION_FILE" ]; then
            log_msg "Found submission file at: $SUBMISSION_FILE"
            if cp "$SUBMISSION_FILE" ${SUBMISSION_DIR}/submission.csv 2>>"${DEBUG_LOG}"; then
                log_msg "Successfully copied submission to ${SUBMISSION_DIR}/submission.csv"
                return 0
            else
                log_msg "ERROR: Failed to copy submission file"
            fi
        fi
        
        # Try to find any submission CSV files
        SUBMISSION_WITH_ID=$(find ${AGENT_DIR}/workspaces -type f \( -name "submission_*.csv" -o -path "*/submission/*.csv" \) 2>/dev/null | sort -r | head -1)
        if [ ! -z "$SUBMISSION_WITH_ID" ]; then
            log_msg "Found submission file with node ID or in submission directory: $SUBMISSION_WITH_ID"
            if cp "$SUBMISSION_WITH_ID" ${SUBMISSION_DIR}/submission.csv 2>>"${DEBUG_LOG}"; then
                log_msg "Successfully copied submission file to ${SUBMISSION_DIR}/submission.csv"
                return 0
            else
                log_msg "ERROR: Failed to copy submission file with ID"
            fi
        fi
        
        log_msg "No submission.csv found in workspace. Listing workspace structure:"
        find ${AGENT_DIR}/workspaces -type f -name "*.csv" 2>/dev/null | head -10 | while read line; do log_msg "  CSV file: $line"; done
        find ${AGENT_DIR}/workspaces -type d \( -name "*submission*" -o -name "*best*" \) 2>/dev/null | head -10 | while read line; do log_msg "  Directory: $line"; done
    else
        log_msg "Workspaces directory not found: ${AGENT_DIR}/workspaces"
    fi
    
    # Also check log directory for best solution (as fallback)
    if [ -d "${AGENT_DIR}/logs" ] && [ ! -f "${SUBMISSION_DIR}/submission.csv" ]; then
        log_msg "Checking log directory for best solution..."
        BEST_SUBMISSION_IN_LOG=$(find ${AGENT_DIR}/logs -type f -path "*/best_solution/submission/submission.csv" 2>/dev/null | sort -r | head -1)
        if [ ! -z "$BEST_SUBMISSION_IN_LOG" ]; then
            log_msg "Found best submission in log directory: $BEST_SUBMISSION_IN_LOG"
            if cp "$BEST_SUBMISSION_IN_LOG" ${SUBMISSION_DIR}/submission.csv 2>>"${DEBUG_LOG}"; then
                log_msg "Successfully copied best submission from log directory"
                return 0
            else
                log_msg "ERROR: Failed to copy best submission from log directory"
            fi
        fi
    fi
    
    # Final check
    if [ -f "${SUBMISSION_DIR}/submission.csv" ]; then
        log_msg "Submission file successfully prepared at ${SUBMISSION_DIR}/submission.csv"
        ls -lh ${SUBMISSION_DIR}/submission.csv >> "${DEBUG_LOG}" 2>&1 || true
    else
        log_msg "WARNING: No submission file was copied to ${SUBMISSION_DIR}/submission.csv"
        log_msg "Listing contents of ${SUBMISSION_DIR}:"
        ls -la ${SUBMISSION_DIR}/ >> "${DEBUG_LOG}" 2>&1 || true
    fi
    log_msg "=== Finished copy_submission function ==="
}

trap cleanup EXIT
trap copy_submission EXIT

# CPU allocation (use defaults, can be overridden via env vars or kwargs passed to Python)
# These can come from env_vars in config.yaml or from Python kwargs (which we'll parse from $@)
# Default to 22 CPUs per run (can be overridden)
start_cpu=${start_cpu:-0}
CPUS_PER_TASK=${cpus_per_task:-22}
# Also try to extract from Python kwargs if passed as start_cpu=X cpus_per_task=Y
for arg in "$@"; do
    if [[ $arg == start_cpu=* ]]; then
        start_cpu=${arg#start_cpu=}
    fi
    if [[ $arg == cpus_per_task=* ]]; then
        CPUS_PER_TASK=${arg#cpus_per_task=}
    fi
done
end_cpu=$((start_cpu + CPUS_PER_TASK - 1))

# Run ML-Master with timeout - capture exit code properly
cd ${AGENT_DIR}
set +e  # Don't exit on error, we want to continue to copy submission
CUDA_VISIBLE_DEVICES=$MEMORY_INDEX taskset -c ${start_cpu}-${end_cpu} timeout $TIME_LIMIT_SECS python main_mcts.py \
  dataset_dir="${dataset_dir}" \
  data_dir="${data_dir}" \
  desc_file="${desc_file}" \
  exp_name="${EXP_ID}_mcts_comp_validcheck_[cpu-${start_cpu}-${end_cpu}]" \
  start_cpu_id="${start_cpu}" \
  cpu_number="${CPUS_PER_TASK}" \
  $@  # Forward all kwargs from config.yaml

EXIT_CODE=$?
set -e  # Re-enable exit on error for the rest of the script

if [ $EXIT_CODE -eq 124 ]; then
  echo "ML-Master timed out after $TIME_LIMIT" >&2
fi

# Force output - write directly to files and stderr
DEBUG_LOG="/home/logs/copy_submission_debug.log"
echo "========== ML-Master Python process finished with exit code: $EXIT_CODE ==========" | tee -a "${DEBUG_LOG}" >&2
echo "========== Starting submission copy process ==========" | tee -a "${DEBUG_LOG}" >&2

# Always copy submission file, regardless of exit code
# This must run after Python finishes, so we do it explicitly here
# The trap will also call it on exit, but this ensures it happens
copy_submission

echo "========== Submission copy process completed ==========" | tee -a "${DEBUG_LOG}" >&2
echo "Final exit code will be: $EXIT_CODE" | tee -a "${DEBUG_LOG}" >&2

exit $EXIT_CODE
