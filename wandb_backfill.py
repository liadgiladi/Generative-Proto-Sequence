"""Backfill test-set results into wandb from slurm logs.

Runs that finished before the summary fix never sent their final test metrics to wandb:
``eval_with_pretrained_model`` logged them with ``add_scalar(tag, value)`` and no step, so
tensorboard recorded them at step 0 while the run's counter was already at 500k-1M, and
wandb's non-decreasing-step rule dropped them. The numbers exist only in the slurm logs,
which is why every result had to be recovered by grepping.

This reads those logs, extracts the printed test numbers, and writes them into the matching
wandb run's summary under exactly the keys the code now uses -- so old and new runs line up
in the same columns in the runs table.

    # see what it would do, change nothing
    poetry run python wandb_backfill.py

    # actually write
    poetry run python wandb_backfill.py --apply

Matching is by wandb run id, parsed from the "Find logs at: wandb/run-<stamp>-<id>/logs"
line each run prints -- unambiguous, unlike the display name.
"""

from __future__ import annotations

import argparse
import glob
import re
from typing import Optional

# "test dataset with critic mean success: 0.582"
# "test dataset with critic-deterministic inference mean success: 0.511"
#
# The printed prefix is f"test dataset{' with critic'}{'-deterministic inference'|''}",
# and the tag the code logs under is f"test-eval{'-with-critic'}{'-deterministic'|''}".
TEST_LINE = re.compile(
    r"^test dataset with critic(?P<det>-deterministic inference)? "
    r"(?P<metric>mean [a-z ]+?): (?P<value>-?[\d.]+)$",
    re.MULTILINE,
)

# Printed phrase -> the metric name used in add_scalar / wandb summary.
METRIC_NAMES = {
    "mean reward": "mean_episodic_return",
    "mean success": "mean_episodic_success",
    "mean num decoder generations": "mean_num_decoder_generations",
    "mean num valid actions in sequences per episode": "mean_num_valid_actions_per_sequence_in_episode",
    "mean num unused actions in sequences per episode": "mean_num_unused_actions_per_sequence_in_episode",
    "mean agent step ratio in sequences per episode": "mean_agent_step_ratio_per_sequence_in_episode",
}

RUN_ID = re.compile(r"wandb/run-\d{8}_\d{6}-(?P<id>[a-z0-9]+)/logs")
RUN_NAME = re.compile(r"View run (?P<name>\S+) at:")


def parse_log(path: str) -> tuple[Optional[str], Optional[str], dict]:
    """Return (wandb run id, display name, {metric key: value}) for one slurm log."""
    try:
        with open(path, "r", errors="replace") as handle:
            text = handle.read()
    except OSError:
        return None, None, {}

    run_id_match = RUN_ID.search(text)
    run_name_match = RUN_NAME.search(text)

    metrics: dict[str, float] = {}
    for match in TEST_LINE.finditer(text):
        metric = METRIC_NAMES.get(match.group("metric").strip())
        if metric is None:
            continue
        tag = "test-eval-with-critic" + ("-deterministic" if match.group("det") else "")
        metrics[f"{tag}/{metric}"] = float(match.group("value"))

    return (
        run_id_match.group("id") if run_id_match else None,
        run_name_match.group("name") if run_name_match else None,
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="asisida-ben-gurion-university-of-the-negev")
    parser.add_argument("--project", default="grid-experiments")
    parser.add_argument("--logs", nargs="*",
                        default=["gps_maze_job-*.out", "gps_maze_nondet_job-*.out",
                                 "sbatch/logs/*.out"],
                        help="glob patterns for slurm logs")
    parser.add_argument("--apply", action="store_true",
                        help="write to wandb; without it, only report what would change")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.logs for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"no logs matched {args.logs} -- run from the repository root")

    print(f"scanning {len(paths)} logs\n")

    found = []
    for path in paths:
        run_id, run_name, metrics = parse_log(path)
        if not metrics:
            continue
        if run_id is None:
            print(f"SKIP {path}: has test results but no wandb run id "
                  f"(run was probably launched without --track)")
            continue
        found.append((path, run_id, run_name, metrics))

    if not found:
        raise SystemExit("no logs contained final test results")

    api = None
    if args.apply:
        import wandb
        api = wandb.Api()

    for path, run_id, run_name, metrics in found:
        success = metrics.get("test-eval-with-critic/mean_episodic_success")
        print(f"{run_name or run_id:<58} success={success}  ({len(metrics)} metrics)  <- {path}")

        if not args.apply:
            continue

        try:
            run = api.run(f"{args.entity}/{args.project}/{run_id}")
            run.summary.update(metrics)
            run.summary.update()  # flush
            print(f"    written to {run.name}")
        except Exception as exc:
            print(f"    FAILED: {exc}")

    if not args.apply:
        print(f"\n{len(found)} runs would be updated. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
