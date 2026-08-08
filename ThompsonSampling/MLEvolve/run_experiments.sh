#!/bin/bash
# run_experiments.sh
# Runs 10 full passes over kaggle_short_list.txt.
# Each pass launches all 9 competitions in parallel (9 × 22 CPUs = 198 CPUs).
# After each pass, generates metadata.json and runs mlebench grading.
#
# Usage: bash run_experiments.sh
#
# All paths have repo-relative defaults and can be overridden via env vars, e.g.:
#   DATASET_DIR=/data/mle-bench MLEBENCH_DIR=/opt/mle-bench \
#   COMP_LIST=/path/to/split.txt RUNS_DIR=/mnt/extra/runs \
#   bash run_experiments.sh
# Other overridable vars: ORCH_LOG_DIR, NUM_RUNS, CPUS_PER_TASK, BASE_SERVER_ID.

set -uo pipefail

# ═══════════════════════════════════════════════════════════════
#  PATHS — resolved relative to this script; override any via env var
# ═══════════════════════════════════════════════════════════════
ROOT="$(cd "$(dirname "$0")" && pwd)"                 # …/ThompsonSampling/MLEvolve
REPO_ROOT="$(cd "${ROOT}/../.." && pwd)"              # repository root
# Prefer Python deps from an .venv if present (see README / pip install -r requirements_*.txt)
for _venv in "${REPO_ROOT}/.venv" "${ROOT}/.venv"; do
  if [[ -x "${_venv}/bin/python" ]]; then
    export PATH="${_venv}/bin:${PATH}"
    break
  fi
done

# Prepared data from `mlebench prepare`.
DATASET_DIR="${DATASET_DIR:-$HOME/.cache/mle-bench/data}"
# mle-bench checkout (submodule at the repo root by default).
MLEBENCH_DIR="${MLEBENCH_DIR:-${REPO_ROOT}/mle-bench}"
# Competition list; defaults to the short list shipped with mle-bench.
COMP_LIST="${COMP_LIST:-${MLEBENCH_DIR}/experiments/splits/kaggle_short_list.txt}"
# Writable output locations; default under this script's directory.
# Point these at a large scratch disk (e.g. RUNS_DIR=/mnt/extra/runs) if desired.
RUNS_DIR="${RUNS_DIR:-${ROOT}/experiment_runs}"
ORCH_LOG_DIR="${ORCH_LOG_DIR:-${RUNS_DIR}/orchestration_logs}"

# ═══════════════════════════════════════════════════════════════
#  FIXED SETTINGS
# ═══════════════════════════════════════════════════════════════
NUM_RUNS="${NUM_RUNS:-10}"
CPUS_PER_TASK="${CPUS_PER_TASK:-22}"
BASE_SERVER_ID="${BASE_SERVER_ID:-100}"   # tasks get server IDs 100–108

# ═══════════════════════════════════════════════════════════════
#  PREFLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════
if [ -z "$DATASET_DIR" ]; then
    echo "ERROR: DATASET_DIR is empty. Unset it to use the default (\$HOME/.cache/mle-bench/data)"
    echo "       or export a valid path, e.g. export DATASET_DIR=/data/mle-bench"
    exit 1
fi

if [ ! -d "$MLEBENCH_DIR" ]; then
    echo "ERROR: mle-bench checkout not found at: ${MLEBENCH_DIR}"
    echo "       Override with: export MLEBENCH_DIR=/path/to/mle-bench"
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY is not set. Export it before running, e.g.:"
    echo "  export OPENAI_API_KEY=sk-..."
    exit 1
fi
export OPENAI_API_KEY

if [ ! -f "$COMP_LIST" ]; then
    echo "ERROR: Competition list not found at: ${COMP_LIST}"
    exit 1
fi

mkdir -p "$RUNS_DIR" "$ORCH_LOG_DIR"

# Read competitions, skipping blank lines and comments
mapfile -t COMPETITIONS < <(grep -v '^\s*$' "$COMP_LIST" | grep -v '^\s*#')
NUM_COMPS=${#COMPETITIONS[@]}
echo "Loaded ${NUM_COMPS} competitions:"
for c in "${COMPETITIONS[@]}"; do echo "  - ${c}"; done
echo ""

# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP  (10 runs × 9 competitions)
# ═══════════════════════════════════════════════════════════════
for run_idx in $(seq 1 $NUM_RUNS); do
    RUN_TAG="run_$(printf '%02d' $run_idx)"
    echo "══════════════════════════════════════════════════════════════"
    echo "  ${RUN_TAG} / run_$(printf '%02d' $NUM_RUNS)  —  started $(date)"
    echo "══════════════════════════════════════════════════════════════"

    run_group_dir="${RUNS_DIR}/${RUN_TAG}"
    mkdir -p "$run_group_dir"

    # ── Launch all competitions in parallel ──
    pids=()
    for task_idx in "${!COMPETITIONS[@]}"; do
        comp="${COMPETITIONS[$task_idx]}"
        start_cpu=$((task_idx * CPUS_PER_TASK))
        server_id=$((BASE_SERVER_ID + task_idx))
        task_log="${ORCH_LOG_DIR}/${RUN_TAG}_${comp}.log"

        bash "$ROOT/run_single_task.sh" \
            "$comp" "$DATASET_DIR" "$server_id" "$start_cpu" "$run_group_dir" \
            > "$task_log" 2>&1 &

        pids+=($!)
        echo "  Launched: ${comp}  cpu_start=${start_cpu}  server=${server_id}  pid=${pids[-1]}"
    done

    # ── Wait for all 9 tasks to finish ──
    echo ""
    for i in "${!pids[@]}"; do
        comp="${COMPETITIONS[$i]}"
        if wait "${pids[$i]}"; then
            echo "  [OK]   ${comp}"
        else
            echo "  [WARN] ${comp} exited with an error — check ${ORCH_LOG_DIR}/${RUN_TAG}_${comp}.log"
        fi
    done

    # ── Post-processing: metadata.json ──
    echo ""
    echo "  Post-processing ${RUN_TAG} ..."

    run_ids=()
    for comp in "${COMPETITIONS[@]}"; do
        # run.py wrote directly into run_group_dir — find the most recently created dir.
        run_dir=$(ls -td "${run_group_dir}/${comp}_"* 2>/dev/null | head -1)

        if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
            echo "  [WARN] No run directory found for ${comp} in ${run_group_dir} — skipping"
            continue
        fi

        exp_name=$(basename "$run_dir")
        run_ids+=("\"${exp_name}\"")
        echo "  Registered: ${exp_name}"
    done

    # Write metadata.json
    run_ids_json=$(printf '%s,' "${run_ids[@]}")
    run_ids_json="[${run_ids_json%,}]"
    echo "{\"runs\": ${run_ids_json}}" > "${run_group_dir}/metadata.json"
    echo "  Written: ${run_group_dir}/metadata.json"

    # ── mlebench evaluation ──
    echo ""
    echo "  Running make_submission.py ..."
    python "${MLEBENCH_DIR}/experiments/make_submission.py" \
        --metadata  "${run_group_dir}/metadata.json" \
        --output    "${run_group_dir}/submission.jsonl" \
        --rel-log-path "logs/MLEvolve.log"

    echo "  Running mlebench grade ..."
    mlebench grade \
        --submission "${run_group_dir}/submission.jsonl" \
        --output-dir "${run_group_dir}"

    echo ""
    echo "  ${RUN_TAG} complete — $(date)"
    echo ""
done

echo "══════════════════════════════════════════════════════════════"
echo "  All ${NUM_RUNS} runs complete."
echo "══════════════════════════════════════════════════════════════"
