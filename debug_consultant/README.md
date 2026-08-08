# Debug Consultant

The debug consultant gives an ML agent a memory of its own bugs. When a node crashes, the
consultant records the error, distills it into a short rule, and injects that rule into later
draft / improve / debug prompts — so the agent stops re-making the same mistake on a different
branch. The store is plain markdown under `<log_dir>/bug_consultant/`
(`BUG_INDEX.md`, `world_model_LATEST.md`, `distilled_guidance.md`).

## Layout

```
debug_consultant/
  README.md
  competitions_9.txt           the 9-task set

  aide/                        the AIDE agent, with the consultant built in
  mlmaster/                    the ML-Master agent, with the consultant built in

  mle-bench/agents/aideml/     tells MLE-bench how to build and run AIDE
  mle-bench/agents/mlmaster/   tells MLE-bench how to build and run ML-Master
```

`aide/` and `mlmaster/` are the agents themselves. The two `mle-bench/agents/*` folders are
the standard MLE-bench agent contract — `Dockerfile` (build the image), `config.yaml`
(register the agent and its arms), `start.sh` (container entrypoint: assemble the task
prompt, launch the agent, sync its outputs back), `additional_notes.txt` (budget text
appended to the prompt), and for AIDE a `config/container_config.json` with the CPU and shared
memory settings. Every agent in the submodule has these; this is not a copy of the harness.

| Agent | Consultant | Hooks | Config |
| --- | --- | --- | --- |
| AIDE | `aide/aide/bug_consultant.py` | `aide/aide/agent.py` | `aide/aide/utils/config.yaml` |
| ML-Master | `mlmaster/agent/bug_consultant.py` | `mlmaster/agent/mcts_agent.py` | `mlmaster/utils/config_mcts.yaml` |

## Install

Every path in this folder is the path it goes to. Copy each piece to the matching location at
the repo root — the two agents become siblings of `mle-bench/`, and the two agent-contract
folders go inside the `mle-bench` submodule:

```bash
cd <repo root>          # the dir containing mle-bench/ and debug_consultant/

mkdir -p aide mlmaster mle-bench/agents/aideml mle-bench/agents/mlmaster
cp -r debug_consultant/aide/.           ./aide/
cp -r debug_consultant/mlmaster/.       ./mlmaster/
cp -r debug_consultant/mle-bench/agents/aideml/.    mle-bench/agents/aideml/
cp -r debug_consultant/mle-bench/agents/mlmaster/.  mle-bench/agents/mlmaster/
```

Resulting layout — this is what the Dockerfiles expect, since both build from the **parent**
context and pull the agent source and the wrapper from two different places:

```
<repo root>/
  aide/                      <- AIDE source
  mlmaster/                  <- ML-Master source
  mle-bench/
    agents/aideml/           <- registers agent-id `aideml/...`
    agents/mlmaster/         <- registers agent-id `mlmaster`
```

Check that MLE-bench sees them:

```bash
cd mle-bench
python -c "from agents.registry import registry; print(registry.get_agent('mlmaster').name)"
```

## Turning it on and off

One line each, under `agent.search`:

```yaml
use_bug_consultant: true      # false = off
```

- AIDE — `aide/aide/utils/config.yaml`, line 66
- ML-Master — `mlmaster/utils/config_mcts.yaml`, line 78

Or pick a preset arm at launch instead of editing the file:

| agent-id | Condition |
| --- | --- |
| `aideml/debug-consultant-on` | consultant on (treatment) |
| `aideml/debug-consultant-off` | consultant off (baseline) |
| `aideml/debug-consultant-no-injection` | records bugs but does not inject them across branches (`inject_shared_context: false`) |
| `aideml/dev-on` / `aideml/dev-off` | smoke-test arms — 8 steps, 1 h limit |
| `mlmaster` | consultant on, `gpt-5-mini` |
| `mlmaster/gpt-5` | consultant on, code model `gpt-5` (feedback stays `gpt-5-mini`) |

For an ML-Master baseline, flip the config key above — it ships no off arm. Its no-injection
ablation was a separately patched image and is not included here.

## Running

Both agents build from a **parent build context** — `aide/` and `mlmaster/` must sit as
siblings of `mle-bench/`, which is exactly the layout above. From `mle-bench/`:

```bash
export OPENAI_API_KEY=...
bash build_agent.sh aideml            # or: bash build_agent.sh mlmaster

python run_agent.py \
  --agent-id aideml/debug-consultant-on \
  --competition-set ../debug_consultant/competitions_9.txt \
  --container-config agents/aideml/config/container_config.json \
  --n-seeds 10 --n-workers 9
```

Config used for the reported runs: `gpt-5-mini`, 2 h per run (`TIME_LIMIT_SECS=7200`), and the
22-CPU / 4 GB-shm container config passed above (it applies to both agents). For a smoke test
use `--n-seeds 1 --n-workers 1` on one competition.

**Check it's actually on:** `<log_dir>/bug_consultant/BUG_INDEX.md` exists when enabled and is
absent when disabled.

### One asymmetry to be aware of

`mle-bench/agents/mlmaster/start.sh` patches `playground-series-s5e12`'s `grade.py` at
container start to score with `roc_auc_score` instead of `accuracy_score`, matching that
competition's stated metric. AIDE's `start.sh` does not. This affects only the in-run
validation server the agent queries, not final grading — but on that one competition the two
agents see different validation feedback. It is left as-is deliberately: changing AIDE's
`start.sh` would make this copy differ from the one that produced AIDE's results.
