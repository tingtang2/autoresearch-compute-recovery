# ML-Master adversarial-EDA experiment

This guide reproduces the ML-Master adversarial-EDA ablation used in this
repository. The committed `utils/data_preview.py` is intentionally **vanilla**.
The treatment injects a fixed, adversarial EDA-memory string into every textual
data preview; the control does not.

The treatment/control comparison is valid only when all other settings are held
constant: competition set, seed(s), data snapshot, model and endpoint, model
parameters, wall-clock budget, CPU/GPU allocation, and ML-Master configuration.
In particular, use the same `run.sh` configuration for both conditions and change
only whether `implement_EDA.py` has patched `utils/data_preview.py`.

## Prerequisites

Complete the environment and data setup in the [ML-Master README](README.md),
including the MLE-bench environment and prepared competition data. Run the
commands below from this directory:

```bash
cd Adversarial_EDA/ML-Master
export OPENAI_API_KEY=...                    # never commit a real key
export DATASET_DIR=/path/to/mle-bench/data   # parent of <competition>/prepared/
export TIME_LIMIT_SECS=7200                  # use the identical value for both arms
```

Configure the model, endpoint, and hardware settings in `run.sh` once, before
starting either arm. Do not change them between the control and treatment runs.

`launch_server.sh` currently has its own `dataset_dir=/path/to/mle-bench` setting
near the top of the file. Set it to the same data root as `DATASET_DIR` before
launching the server. Start it once per machine/session; it continues in the
background and writes `grading_server.out`.

```bash
bash launch_server.sh
```

## Control: no EDA injection

First verify that the checked-in preview implementation is clean. `--check` is
read-only; the regression script also verifies that patching and reverting round
trip exactly.

```bash
python test_data_preview_clean.py
python implement_EDA.py --check
# Expected status: vanilla (not patched)

bash run.sh spaceship-titanic 42
```

Archive or otherwise label the resulting `logs/` and `workspaces/` output as the
**control** condition before running the treatment. For a multi-competition study,
run the same split and seed list for each arm.

## Treatment: adversarial EDA injection

Apply the patch, confirm its state, and rerun the exact same workload. The patch
adds the canonical text from `../eda_findings.txt`, which is also used by the
LLM-as-a-judge analysis.

```bash
python implement_EDA.py
python implement_EDA.py --check
# Expected status: already EDA-enabled

bash run.sh spaceship-titanic 42
```

Do not restart with different seeds or modify `run.sh` between these commands.
The only intentional difference from the control is the injected preview text.

## Batch runs

For either condition, use the same split file, number of seeds, parallelism, and
environment variables. Apply the patch only before the treatment batch.

```bash
# Control: first confirm `python implement_EDA.py --check` reports vanilla.
DATASET_DIR=/path/to/mle-bench/data \
bash run_batch_parallel.sh ../../mle-bench/experiments/splits/dev.txt 3 12

# After archiving/labeling the control output, enable the treatment and repeat
# the identical command.
python implement_EDA.py
DATASET_DIR=/path/to/mle-bench/data \
bash run_batch_parallel.sh ../../mle-bench/experiments/splits/dev.txt 3 12
```

The batch launcher preserves prior output by default. If you use
`CLEAN_WORKSPACES=1`, archive the previous arm first: it intentionally removes
the prior `workspaces/run` and `logs/run` contents.

## Restore the clean checkout

Always return the source tree to the vanilla control state before committing or
starting an unrelated run:

```bash
python implement_EDA.py --revert
python test_data_preview_clean.py
```

`implement_EDA.py` writes a `utils/data_preview.py.bak` backup by default. The
patcher is idempotent, so applying it twice does not duplicate the injection.

## Analyze the runs

Once the control and treatment logs are collected, use the
[LLM-as-a-judge analysis guide](../llm-as-a-judge/README.md). It recognizes
ML-Master's `ml-master.verbose.log` files and evaluates the same injected text.
