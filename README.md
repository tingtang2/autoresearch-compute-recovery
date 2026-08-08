# Recovering Wasted Compute in Autoresearch Agents

This repository contains the released code supporting our COLM 2026 paper, *Recovering Wasted Compute in Autoresearch Agents*.

The experiments study how several agent interventions affect MLE-bench performance:

- Thompson Sampling for search/backtracking
- adversarial exploratory-data-analysis (EDA) context
- a debugging consultant with cross-branch bug memory
- prompt- and control-loop-based hyperparameter optimization (HPO)

This release includes the MLE-bench harness in [`mle-bench/`](mle-bench) alongside the experiment code. The individual experiment directories contain the agent implementations and the exact commands used to run the corresponding treatment and control conditions.

## Repository layout

| Directory | Contents |
| --- | --- |
| [`ThompsonSampling/`](ThompsonSampling) | Thompson-Sampling experiments for AIDE and MLEvolve. |
| [`Adversarial_EDA/`](Adversarial_EDA) | Adversarial-EDA experiments for AIDE and ML-Master. |
| [`debug_consultant/`](debug_consultant) | Debug-consultant experiments for AIDE and ML-Master. |
| [`hpo/`](hpo) | 2x2 HPO factorial for AIDE and ML-Master. |
| [`aideml/`](aideml) | AIDE runtime used by the standalone AIDE Thompson-Sampling and EDA experiments. |
| [`mle-bench/`](mle-bench) | MLE-bench evaluation harness, data preparation tools, and container environment. |

## Quick start

Clone the repository:

```bash
git clone https://github.com/tingtang2/autoresearch-compute-recovery.git
cd autoresearch-compute-recovery
```

MLE-bench requires Python 3.11 or newer, Docker, Kaggle credentials, and substantial compute and storage for many competitions. Install the harness in a dedicated environment:

```bash
python3.11 -m venv .venv-mlebench
source .venv-mlebench/bin/activate
python -m pip install --upgrade pip
python -m pip install -e mle-bench

cd mle-bench
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
mlebench prepare -c <competition-id>
cd ..
```

`mlebench prepare` downloads competition data through the Kaggle API. Configure `~/.kaggle/kaggle.json` first, and use the [MLE-bench README](mle-bench/README.md) for dataset, runtime, and container requirements.

Each experiment has its own dependencies and may require a separate Python environment. Export only the API key required by the agent you intend to run, for example:

```bash
export OPENAI_API_KEY=...
```

> **Isolation note:** the Debug Consultant and HPO instructions install agent wrappers into `mle-bench/agents/`. Some wrappers currently use the same destination names, so run each experiment family in a separate working copy of this repository or Git worktree. Do not install all experiment families into one shared `mle-bench/agents/` tree.

## Reproducing experiments

Start with the linked experiment README. It defines the relevant treatment/control pair, model settings, task set, and run command.

| Experiment | What varies | Code and commands |
| --- | --- | --- |
| AIDE Thompson Sampling | Thompson node selection vs. random selection at backtracking points | [AIDE Thompson Sampling README](ThompsonSampling/AIDE/README_AIDE.MD) |
| MLEvolve Thompson Sampling | MLEvolve search and backtracking experiments | [MLEvolve README](ThompsonSampling/MLEvolve/README_TS.MD) |
| AIDE adversarial EDA | An injected EDA finding vs. a clean no-injection control | [AIDE adversarial EDA README](Adversarial_EDA/AIDE/README.md) |
| ML-Master adversarial EDA | An injected EDA finding vs. a clean no-injection control | [ML-Master adversarial EDA guide](Adversarial_EDA/ML-Master/EDA_EXPERIMENT.md) |
| Debug consultant | Bug-memory consultant enabled vs. disabled or no-injection ablations | [Debug Consultant README](debug_consultant/README.md) |
| HPO factorial | Prompt HPO and control-loop HPO, independently and jointly enabled | [HPO README](hpo/README.md) |

### Standalone AIDE experiments

The AIDE Thompson-Sampling and adversarial-EDA experiments use the root-level `aideml` package rather than an MLE-bench agent wrapper:

```bash
python3 -m venv .venv-aide
source .venv-aide/bin/activate
python -m pip install -e aideml
```

Then follow the treatment/control commands in the [AIDE Thompson-Sampling README](ThompsonSampling/AIDE/README_AIDE.MD) or the [AIDE adversarial-EDA README](Adversarial_EDA/AIDE/README.md).

For a clean EDA control, verify the injection guard before launching a run:

```bash
python Adversarial_EDA/AIDE/test_eda_hook.py
```

### Containerized MLE-bench experiments

The Debug Consultant and HPO experiments use MLE-bench agent contracts: a Dockerfile, `config.yaml`, `start.sh`, and agent source. Their READMEs describe how to install those contracts into an isolated copy of the repository's `mle-bench/agents/` directory, build the images, and invoke `run_agent.py`.

- Debug Consultant: [installation and run instructions](debug_consultant/README.md#install)
- HPO: [AIDE instructions](hpo/aide/README.md) and [ML-Master instructions](hpo/mlmaster/README.md)

## Reproducibility notes

- Use the same competition set, seeds, model, API endpoint, step budget, and time budget across a treatment/control pair. The experiment READMEs identify the intervention that is intended to differ.
- MLE-bench data, Docker images, model access, and API calls are not included in this repository. They must be provisioned by the person running an experiment.
- Agent-generated code can execute arbitrary commands inside its runtime. Follow the MLE-bench container-isolation guidance and do not expose unnecessary credentials or host directories.
- Some commands are syntax- and wiring-validated but have not been run end-to-end in this checkout. A full reproduction requires prepared competition data, the documented compute environment, and valid model credentials.

## License and provenance

This repository combines code from several upstream projects. The top-level [MIT License](LICENSE) applies only to this project's original contributions; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and each component's local license for the terms that apply to included third-party code. Some included agent implementations require separate provenance and licensing review.

The primary upstream projects modified or integrated here are:

- [MLE-bench](https://github.com/openai/mle-bench)
- [AIDE](https://github.com/wecoai/aideml)
- [MLEvolve](https://github.com/InternScience/MLEvolve)
- [ML-Master](https://github.com/sjtu-sai-agents/ML-Master)

## Citation

Citation information for the accompanying paper will be added with the public paper release.
