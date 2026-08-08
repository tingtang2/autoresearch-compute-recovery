"""Launch AIDE with the adversarial EDA memory injected.

This applies ``eda_hook.patch_agent()`` (which reads the ``EDA_FINDINGS`` env var)
*before* importing/using AIDE, then hands off to a normal AIDE entrypoint. All
extra command-line arguments are forwarded to AIDE unchanged (AIDE parses them via
OmegaConf ``from_cli``), e.g. ``data_dir=... desc_file=... exp_name=...``.

Baseline vs. injected is controlled entirely by ``EDA_FINDINGS``:
  * unset/empty  -> patch is a no-op, AIDE runs as the clean baseline
  * set          -> the finding is injected into the agent's Memory prompt

Choose the search agent with ``--agent``:
  * baseline (default) -> aide.run           (no MCTS)
  * backtrack          -> aide.run_backtrack (MCTS / Thompson Sampling)

Examples:
    # Injected EDA, baseline search agent:
    EDA_FINDINGS="target y is imbalanced (88% / 12%); duration corr strongest" \
        python run_with_eda.py -- data_dir=/data/comp/prepared/public \
            desc_file=/data/comp/full_instructions.txt exp_name=eda_injected

    # Injected EDA + Thompson Sampling:
    EDA_FINDINGS="..." python run_with_eda.py --agent backtrack -- \
        data_dir=... desc_file=... exp_name=eda_mcts agent.search.use_mcts=true
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--agent",
        choices=("baseline", "backtrack"),
        default="baseline",
        help="Which AIDE entrypoint to run (default: baseline / no MCTS).",
    )
    # Everything after `--` (or any unrecognised token) is forwarded to AIDE.
    args, aide_args = parser.parse_known_args()

    # Make eda_hook importable regardless of the caller's working directory.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eda_hook import patch_agent

    patch_agent()  # no-op if EDA_FINDINGS is unset/empty

    # AIDE reads its config from sys.argv via OmegaConf.from_cli(); rebuild argv so
    # only the forwarded key=value args remain.
    sys.argv = [sys.argv[0]] + [a for a in aide_args if a != "--"]

    if args.agent == "backtrack":
        from aide.run_backtrack import run
    else:
        from aide.run import run

    run()


if __name__ == "__main__":
    main()
