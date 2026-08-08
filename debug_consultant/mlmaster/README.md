# ML-Master (vendored, with the debug consultant)

This directory is a vendored copy of **ML-Master** by [SJTU SAI](https://sai.sjtu.edu.cn/),
modified to add the debug consultant.

- Upstream project: https://sjtu-sai-agents.github.io/ML-Master
- Paper: https://arxiv.org/abs/2506.16499

Licensing and copyright for everything here except the consultant follow the upstream project.

## What was changed

| File | Change |
| --- | --- |
| `agent/bug_consultant.py` | added — the debug consultant |
| `agent/mcts_agent.py` | records bugs after each node, injects prevention guidance into prompts |
| `utils/config_mcts.yaml` | `agent.search.use_bug_consultant` and related keys |

Turn it on or off with `use_bug_consultant` in `utils/config_mcts.yaml` (line 78). See
`../README.md` for how to install and run this under MLE-bench.

Upstream's standalone instructions (`run.sh`, `launch_server.sh`, DeepSeek setup) are not
included — this copy is launched by `mle-bench/agents/mlmaster/start.sh`, which calls
`main_mcts.py` directly.
