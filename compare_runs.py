"""Side-by-side comparison of runs across the full GPS metric set.

Success rate is not the paper's only claim. GPS is about *sequence-level* decision making,
so the metrics that matter include how many decoder generations an episode needs, how much
of each emitted plan is actually used, how close the agent's path is to optimal, and how
much compute a decision costs. All of these are printed at the end of a run by both the
Transformer tree and the paper's MLP tree -- but scattered across lines, and only the
Transformer tree pushes any of them to wandb.

    poetry run python compare_runs.py sbatch/logs/i2_*.out sbatch/logs/i3_*.out
    poetry run python compare_runs.py --csv all_maps.csv sbatch/logs/j1_*.out sbatch/logs/j2_*.out

Columns are labelled from the "=== J1: map=... ===" header the j-series scripts echo, since
they are parameterised by map and every one of them writes to the same j1_gpt_map-<jobid>.out
filename. Logs without that header fall back to their filename.

Works on logs from either tree, since both print the same lines.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

# "test dataset with critic mean success: 0.742"
# "test dataset with critic-deterministic inference mean success: 0.643"
TEST_METRIC = re.compile(
    r"^test dataset with critic(?P<det>-deterministic inference)? "
    r"(?P<metric>mean [a-z \-]+?): (?P<value>-?[\d.]+)$",
    re.MULTILINE,
)

# Timing lines carry no pass prefix. eval_test_dataset runs the deterministic pass first
# (when deterministic_inference is set) and the stochastic one second, so occurrences are
# taken in that order.
TIMING = re.compile(
    r"^(?P<metric>average wall-clock time per optimal step|"
    r"std wall-clock time per optimal step|"
    r"average inference time per decision \(successful episodes only\)|"
    r"std inference time per decision \(successful episodes only\)"
    r"): (?P<value>-?[\d.e+]+) seconds$",
    re.MULTILINE,
)

# "=== J1: map=16x16_empty  length_weight=0.002  train_pairs=500 ==="
# The j-series scripts are parameterised by map, so every one of them writes to
# j1_gpt_map-<jobid>.out and the filename alone cannot tell two maps apart. Prefer the
# header the script echoes.
HEADER = re.compile(
    r"^=== (?P<arm>J\d): map=(?P<map>\S+)(?:\s+length_weight=(?P<weight>\S+))?", re.MULTILINE
)

SIMPLE = {
    "best val success": re.compile(r"best val mean success=([\d.]+)"),
    "best val step": re.compile(r"best val mean global step=(\d+)"),
    "wall clock (h)": re.compile(r"WALL_CLOCK_HOURS=([\d.]+)"),
    "last step": re.compile(r"global_step=(\d+)"),
    "successful episodes": re.compile(r"successful episodes: (\d+)/\d+"),
    "total decisions": re.compile(r"total decisions in successful episodes: (\d+)"),
}

# Display order and shorter labels. Anything measured but not listed here still appears,
# under its raw name, after these.
ORDER = [
    "mean success",
    "mean reward",
    "mean num decoder generations",
    "mean num valid actions in sequences per episode",
    "mean num unused actions in sequences per episode",
    "mean agent step ratio in sequences per episode",
    "mean num actions per episode",
    "mean global visited position count in sequences per episode",
    "mean prev visited position count in sequences per episode",
    "mean current visited position count in sequences per episode",
]

SHORT = {
    "mean success": "success",
    "mean reward": "return",
    "mean num decoder generations": "decoder generations/ep",
    "mean num valid actions in sequences per episode": "valid actions/seq",
    "mean num unused actions in sequences per episode": "unused actions/seq",
    "mean agent step ratio in sequences per episode": "agent step ratio",
    "mean num actions per episode": "actions/ep",
    "mean global visited position count in sequences per episode": "revisits (global)",
    "mean prev visited position count in sequences per episode": "revisits (prev seq)",
    "mean current visited position count in sequences per episode": "revisits (current seq)",
}


def parse(path: str) -> dict:
    try:
        with open(path, "r", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        return {"_error": str(exc)}

    out: dict[str, str] = {}

    header = HEADER.search(text)
    if header:
        # Strip the shared "16x16_" prefix so columns stay narrow; the arm (J1 = GPT,
        # J2 = MLP) matters more than the resolution, which is in the map name anyway.
        name = header.group("map").replace("16x16_", "").replace("_v1", "")
        arm = "gpt" if header.group("arm") == "J1" else "mlp"
        if header.group("weight") == "0":
            arm += "/plain"       # the no-length-penalty arm, comparable to the MLP's
        out["_label"] = f"{name}:{arm}"

    for match in TEST_METRIC.finditer(text):
        suffix = " (det)" if match.group("det") else ""
        out[match.group("metric").strip() + suffix] = match.group("value")

    # Timing: first occurrence = deterministic pass, second = stochastic.
    seen: dict[str, int] = {}
    for match in TIMING.finditer(text):
        metric = match.group("metric").replace(" (successful episodes only)", "")
        n = seen.get(metric, 0)
        seen[metric] = n + 1
        out[metric + (" (det)" if n == 0 else "")] = match.group("value")

    for label, pattern in SIMPLE.items():
        found = pattern.findall(text)
        if found:
            out[label] = found[-1]

    # A run that died before its final evaluation has no test metrics, and its other
    # numbers are best-checkpoint values from an incomplete run -- flag rather than
    # silently present them as comparable.
    out["ABORTED"] = "yes" if "aborting training at global_step" in text else ""
    if "mean success" not in out:
        out["ABORTED"] = (out["ABORTED"] or "no") + " / no final eval"

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--csv", help="also write the table to this path")
    parser.add_argument("--all", action="store_true",
                        help="include the deterministic-inference pass as well")
    args = parser.parse_args()

    runs = []
    for path in args.logs:
        data = parse(path)
        if "_error" in data:
            continue
        # The map/arm label the script echoed, when there is one -- j1_gpt_map-*.out is
        # the same filename for every map in the sweep.
        fallback = os.path.basename(path).rsplit("-", 1)[0]
        runs.append((data.pop("_label", None) or fallback, data))
    if not runs:
        sys.exit("no readable logs")

    # Several runs of one map/arm -- a 500k and a 1M budget, or a repeated seed -- would
    # otherwise land under identical column headings. Disambiguate by step budget first
    # (that is the difference that matters), then by a seed counter for true repeats.
    counts: dict[str, int] = {}
    for name, _ in runs:
        counts[name] = counts.get(name, 0) + 1
    repeats: dict[str, int] = {}
    for i, (name, data) in enumerate(runs):
        if counts[name] < 2:
            continue
        last = data.get("last step")
        budget = f"@{round(int(last) / 1000):d}k" if last and last.isdigit() else ""
        repeats[name] = repeats.get(name, 0) + 1
        runs[i] = (f"{name}{budget}#{repeats[name]}", data)

    keys = [k for k in ORDER if any(k in d for _, d in runs)]
    keys += [k for k in SIMPLE if any(k in d for _, d in runs)]
    keys += ["average wall-clock time per optimal step",
             "average inference time per decision"]
    if args.all:
        keys += [k for k in sorted({k for _, d in runs for k in d}) if k.endswith("(det)")]
    keys += ["ABORTED"]
    keys = [k for k in dict.fromkeys(keys) if any(d.get(k) for _, d in runs)]

    width = max(len(SHORT.get(k, k)) for k in keys) + 2
    col = max(max(len(n) for n, _ in runs), 12) + 2

    print(f"{'metric':<{width}}" + "".join(f"{n:>{col}}" for n, _ in runs))
    print("-" * (width + col * len(runs)))
    for key in keys:
        label = SHORT.get(key, key)
        print(f"{label:<{width}}" + "".join(f"{d.get(key, '-'):>{col}}" for _, d in runs))

    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric"] + [n for n, _ in runs])
            for key in keys:
                writer.writerow([SHORT.get(key, key)] + [d.get(key, "") for _, d in runs])
        print(f"\nwritten to {args.csv}")


if __name__ == "__main__":
    main()
