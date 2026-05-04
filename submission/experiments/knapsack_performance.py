from __future__ import annotations

import argparse
import csv
import random
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(slots=True)
class Item:
    value: int
    weight: int


@dataclass(slots=True)
class RunResult:
    n: int
    capacity: int
    repetition: int
    algorithm: str
    value: int | None
    used_weight: int | None
    time_seconds: float
    optimum: int | None


def generate_instance(n: int, capacity: int, seed: int) -> list[Item]:
    rng = random.Random(seed)
    return [
        Item(
            value=rng.randint(1, 1000),
            weight=rng.randint(1, max(2, capacity // 20)),
        )
        for _ in range(n)
    ]


def dp_knapsack(items: list[Item], capacity: int, max_cells: int) -> tuple[int | None, int | None, float]:
    cells = len(items) * capacity
    if cells > max_cells:
        return None, None, 0.0

    started = time.perf_counter()
    dp = [0] * (capacity + 1)
    keep: list[list[bool]] = [[False] * (capacity + 1) for _ in items]

    for idx, item in enumerate(items):
        for budget in range(capacity, item.weight - 1, -1):
            candidate = dp[budget - item.weight] + item.value
            if candidate > dp[budget]:
                dp[budget] = candidate
                keep[idx][budget] = True

    best_budget = max(range(capacity + 1), key=lambda budget: dp[budget])
    used_weight = 0
    budget = best_budget
    for idx in range(len(items) - 1, -1, -1):
        item = items[idx]
        if keep[idx][budget]:
            used_weight += item.weight
            budget -= item.weight

    return dp[best_budget], used_weight, time.perf_counter() - started


def greedy_density(items: list[Item], capacity: int) -> tuple[int, int, float]:
    return _greedy(items, capacity, key=lambda item: item.value / item.weight)


def greedy_value(items: list[Item], capacity: int) -> tuple[int, int, float]:
    return _greedy(items, capacity, key=lambda item: item.value)


def greedy_lightest(items: list[Item], capacity: int) -> tuple[int, int, float]:
    return _greedy(items, capacity, key=lambda item: -item.weight)


def random_feasible(items: list[Item], capacity: int, seed: int) -> tuple[int, int, float]:
    started = time.perf_counter()
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    value = 0
    used = 0
    for item in shuffled:
        if used + item.weight <= capacity:
            used += item.weight
            value += item.value
    return value, used, time.perf_counter() - started


def simulated_annealing(items: list[Item], capacity: int, seed: int) -> tuple[int, int, float]:
    started = time.perf_counter()
    rng = random.Random(seed)
    selected = [False] * len(items)
    value = 0
    used = 0

    # Start from a strong feasible solution so SA improves a realistic heuristic baseline.
    for idx, item in sorted(enumerate(items), key=lambda pair: pair[1].value / pair[1].weight, reverse=True):
        if used + item.weight <= capacity:
            selected[idx] = True
            used += item.weight
            value += item.value

    best_selected = selected[:]
    best_value = value
    best_used = used
    temperature = max(1.0, len(items) / 2)
    iterations = min(25_000, max(2_000, len(items) * 12))

    for step in range(iterations):
        idx = rng.randrange(len(items))
        item = items[idx]
        next_used = used - item.weight if selected[idx] else used + item.weight
        if next_used > capacity:
            continue

        delta = -item.value if selected[idx] else item.value
        accept = delta >= 0 or rng.random() < pow(2.718281828, delta / max(temperature, 1e-9))
        if accept:
            selected[idx] = not selected[idx]
            used = next_used
            value += delta
            if value > best_value:
                best_selected = selected[:]
                best_value = value
                best_used = used

        temperature *= 0.9995
        if step % 250 == 0 and step > 0:
            # Small reheating helps escape local regions without making the run nondeterministic.
            temperature = max(0.01, temperature * 1.02)

    return best_value, best_used, time.perf_counter() - started


def _greedy(items: list[Item], capacity: int, key: Callable[[Item], float]) -> tuple[int, int, float]:
    started = time.perf_counter()
    value = 0
    used = 0
    for item in sorted(items, key=key, reverse=True):
        if used + item.weight <= capacity:
            used += item.weight
            value += item.value
    return value, used, time.perf_counter() - started


def run_experiments(output_dir: Path, max_cells: int, repetitions: int) -> list[RunResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = [
        {"n": 100, "capacity": 500},
        {"n": 500, "capacity": 750},
        {"n": 1000, "capacity": 1000},
        {"n": 5000, "capacity": 1500},
        {"n": 10000, "capacity": 2000},
    ]

    rows: list[RunResult] = []
    for setting in settings:
        n = setting["n"]
        capacity = setting["capacity"]
        for repetition in range(repetitions):
            seed = 42 + (n * 1000) + repetition
            items = generate_instance(n=n, capacity=capacity, seed=seed)

            dp_value, dp_weight, dp_time = dp_knapsack(items, capacity, max_cells=max_cells)
            density_value, density_weight, density_time = greedy_density(items, capacity)
            value_value, value_weight, value_time = greedy_value(items, capacity)
            light_value, light_weight, light_time = greedy_lightest(items, capacity)
            anneal_value, anneal_weight, anneal_time = simulated_annealing(items, capacity, seed=seed + 31)
            random_value, random_weight, random_time = random_feasible(items, capacity, seed=seed + 17)

            optimum = dp_value
            rows.extend(
                [
                    RunResult(n, capacity, repetition, "DP exact", dp_value, dp_weight, dp_time, optimum),
                    RunResult(n, capacity, repetition, "Greedy density", density_value, density_weight, density_time, optimum),
                    RunResult(n, capacity, repetition, "Simulated annealing", anneal_value, anneal_weight, anneal_time, optimum),
                    RunResult(n, capacity, repetition, "Greedy value", value_value, value_weight, value_time, optimum),
                    RunResult(n, capacity, repetition, "Greedy lightest", light_value, light_weight, light_time, optimum),
                    RunResult(n, capacity, repetition, "Random feasible", random_value, random_weight, random_time, optimum),
                ]
            )

    summary = summarize(rows)
    write_raw_csv(rows, output_dir / "knapsack_runs.csv")
    write_summary_csv(summary, output_dir / "knapsack_summary.csv")
    write_latex_table(summary, output_dir / "knapsack_summary_table.tex")
    write_latex_algorithm_table(summary, output_dir / "algorithm_comparison_table.tex")
    write_latex_timeout_table(summary, output_dir / "knapsack_timeout_table.tex")
    write_plots(summary, output_dir / "figures")
    mirror_paper_assets(output_dir)
    return rows


def summarize(rows: list[RunResult]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int, str], list[RunResult]] = {}
    for row in rows:
        grouped.setdefault((row.n, row.capacity, row.algorithm), []).append(row)

    summary: list[dict[str, object]] = []
    for (n, capacity, algorithm), group in sorted(grouped.items()):
        solved = [row for row in group if row.value is not None]
        gaps = [
            100.0 * (row.optimum - row.value) / max(1, row.optimum)
            for row in group
            if row.value is not None and row.optimum is not None
        ]
        times = [row.time_seconds for row in solved]
        weights = [row.used_weight or 0 for row in solved]
        values = [row.value or 0 for row in solved]

        summary.append(
            {
                "n": n,
                "capacity": capacity,
                "algorithm": algorithm,
                "runs": len(group),
                "solved_runs": len(solved),
                "timeout_rate": 1.0 - (len(solved) / max(1, len(group))),
                "mean_value": _mean(values),
                "mean_used_weight": _mean(weights),
                "mean_time_ms": 1000.0 * _mean(times),
                "std_time_ms": 1000.0 * _std(times),
                "ci95_time_ms": 1000.0 * _ci95(times),
                "mean_gap_percent": "" if not gaps else _mean(gaps),
                "ci95_gap_percent": "" if not gaps else _ci95(gaps),
                "max_gap_percent": "" if not gaps else max(gaps),
            }
        )
    return summary


def _mean(values: list[float | int]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float | int]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _ci95(values: list[float | int]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * float(statistics.stdev(values)) / (len(values) ** 0.5)


def write_raw_csv(rows: list[RunResult], path: Path) -> None:
    fieldnames = [
        "n",
        "capacity",
        "repetition",
        "algorithm",
        "value",
        "used_weight",
        "time_seconds",
        "optimum",
        "accuracy_gap_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            gap = ""
            if row.value is not None and row.optimum is not None:
                gap = 100.0 * (row.optimum - row.value) / max(1, row.optimum)
            writer.writerow(
                {
                    "n": row.n,
                    "capacity": row.capacity,
                    "repetition": row.repetition,
                    "algorithm": row.algorithm,
                    "value": "TIMEOUT" if row.value is None else row.value,
                    "used_weight": "" if row.used_weight is None else row.used_weight,
                    "time_seconds": row.time_seconds,
                    "optimum": "" if row.optimum is None else row.optimum,
                    "accuracy_gap_percent": gap,
                }
            )


def write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(rows: list[dict[str, object]], path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["n"] in {100, 1000, 10000}
        and row["algorithm"] in {"DP exact", "Greedy density", "Simulated annealing", "Greedy value", "Random feasible"}
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Repeated Synthetic Knapsack Results (100 Runs per Size)}",
        r"\label{tab:repeated-knapsack}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"$N$ & Algorithm & Mean Value & Mean Time (ms) & Mean Gap (\%) & Timeout Rate \\",
        r"\hline",
    ]
    for row in selected:
        gap = row["mean_gap_percent"]
        gap_text = "--" if gap == "" else f"{float(gap):.2f}"
        if int(row["solved_runs"]) == 0:
            value_text = "--"
            time_text = "--"
        else:
            value_text = f"{float(row['mean_value']):.1f}"
            time_text = f"{float(row['mean_time_ms']):.3f}"
        algorithm = str(row["algorithm"])
        display_algorithm = algorithm
        row_prefix = ""
        if algorithm in {"DP exact", "Greedy density"}:
            row_prefix = r"\rowcolor{tokenpackhighlight}"
            display_algorithm = rf"\textbf{{{algorithm}}}"
        lines.append(
            f"{row_prefix}{row['n']} & {display_algorithm} & {value_text} & "
            f"{time_text} & {gap_text} & {float(row['timeout_rate']):.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_algorithm_table(rows: list[dict[str, object]], path: Path) -> None:
    selected = [row for row in rows if row["n"] == 1000]
    order = ["DP exact", "Greedy density", "Simulated annealing", "Greedy lightest", "Greedy value", "Random feasible"]
    selected.sort(key=lambda row: order.index(str(row["algorithm"])))
    dp_time = next(float(row["mean_time_ms"]) for row in selected if row["algorithm"] == "DP exact")
    lines = [
        r"\begin{table}[!t]",
        r"\caption{Paired Statistical Comparison of Algorithms for $N=1000$ over 100 Runs}",
        r"\label{tab:algorithm-comparison}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Algorithm & Mean Value & Mean Time (ms) & Gap (\%) & 95\% CI Gap & Speedup \\",
        r"\hline",
    ]
    for row in selected:
        gap = row["mean_gap_percent"]
        gap_text = "--" if gap == "" else f"{float(gap):.2f}"
        ci_gap = row["ci95_gap_percent"]
        ci_text = "--" if ci_gap == "" else f"{float(ci_gap):.2f}"
        mean_time = float(row["mean_time_ms"])
        speedup = "--" if row["algorithm"] == "DP exact" else f"{dp_time / max(mean_time, 1e-9):.1f}x"
        algorithm = str(row["algorithm"])
        display_algorithm = algorithm
        row_prefix = ""
        if algorithm in {"DP exact", "Greedy density"}:
            row_prefix = r"\rowcolor{tokenpackhighlight}"
            display_algorithm = rf"\textbf{{{algorithm}}}"
        lines.append(
            f"{row_prefix}{display_algorithm} & {float(row['mean_value']):.1f} & {mean_time:.3f} & "
            f"{gap_text} & {ci_text} & {speedup} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}%", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_timeout_table(rows: list[dict[str, object]], path: Path) -> None:
    dp_rows = [row for row in rows if row["algorithm"] == "DP exact"]
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{DP Scalability as $nC$ Grows}",
        r"\label{tab:dp-scalability}",
        r"\centering",
        r"\begin{tabular}{rrrr}",
        r"\hline",
        r"$N$ & Capacity & $nC$ States & Timeout Rate \\",
        r"\hline",
    ]
    for row in dp_rows:
        states = int(row["n"]) * int(row["capacity"])
        lines.append(
            f"{row['n']} & {row['capacity']} & {states} & {float(row['timeout_rate']):.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(rows: list[dict[str, object]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    algorithms = ["DP exact", "Greedy density", "Simulated annealing", "Greedy lightest", "Greedy value", "Random feasible"]
    colors = {
        "DP exact": "#0057b8",
        "Greedy density": "#008a2e",
        "Simulated annealing": "#00838f",
        "Greedy lightest": "#ef6c00",
        "Greedy value": "#c62828",
        "Random feasible": "#6a1b9a",
    }

    _plot_runtime_scaling(rows, algorithms, colors, output_dir / "runtime_scaling.png")
    _plot_gap_comparison(rows, algorithms, colors, output_dir / "gap_comparison_n1000.png")
    _plot_dp_scalability(rows, output_dir / "dp_state_scalability.png")


def mirror_paper_assets(output_dir: Path) -> None:
    paper_dir = output_dir.parent / "paper"
    tables_dir = paper_dir / "tables"
    figures_dir = paper_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "algorithm_comparison_table.tex",
        "knapsack_summary_table.tex",
        "knapsack_timeout_table.tex",
    ]:
        shutil.copy2(output_dir / filename, tables_dir / filename)

    for figure in (output_dir / "figures").glob("*.png"):
        shutil.copy2(figure, figures_dir / figure.name)


def _plot_runtime_scaling(
    rows: list[dict[str, object]],
    algorithms: list[str],
    colors: dict[str, str],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    for algorithm in algorithms:
        selected = [
            row
            for row in rows
            if row["algorithm"] == algorithm and int(row["solved_runs"]) > 0
        ]
        selected.sort(key=lambda row: int(row["n"]))
        ax.plot(
            [int(row["n"]) for row in selected],
            [float(row["mean_time_ms"]) for row in selected],
            marker="o",
            linewidth=3.2 if algorithm in {"DP exact", "Greedy density"} else 1.7,
            markersize=6 if algorithm in {"DP exact", "Greedy density"} else 4,
            label=algorithm,
            color=colors[algorithm],
            alpha=1.0 if algorithm in {"DP exact", "Greedy density"} else 0.65,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of items (N)")
    ax.set_ylabel("Mean runtime (ms, log scale)")
    ax.set_title("Runtime Scaling Across Algorithms")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_gap_comparison(
    rows: list[dict[str, object]],
    algorithms: list[str],
    colors: dict[str, str],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["n"] == 1000 and row["mean_gap_percent"] != ""]
    selected.sort(key=lambda row: algorithms.index(str(row["algorithm"])))
    labels = [str(row["algorithm"]).replace("Greedy ", "G. ") for row in selected]
    gaps = [float(row["mean_gap_percent"]) for row in selected]

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=180)
    bar_colors = [colors[str(row["algorithm"])] for row in selected]
    bar_edges = ["#003f88" if str(row["algorithm"]) in {"DP exact", "Greedy density"} else "#444444" for row in selected]
    bar_widths = [2.2 if str(row["algorithm"]) in {"DP exact", "Greedy density"} else 0.7 for row in selected]
    bars = ax.bar(labels, gaps, color=bar_colors, edgecolor=bar_edges, linewidth=bar_widths)
    ax.set_ylabel("Mean accuracy gap from DP optimum (%)")
    ax.set_title("Approximation Gap at N=1000")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_ylim(0, max(gaps) * 1.15)
    ax.tick_params(axis="x", rotation=18)
    for bar, gap in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(gaps) * 0.025,
            f"{gap:.2f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_dp_scalability(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    dp_rows = [row for row in rows if row["algorithm"] == "DP exact"]
    dp_rows.sort(key=lambda row: int(row["n"]) * int(row["capacity"]))
    solved = [row for row in dp_rows if int(row["solved_runs"]) > 0]
    timed_out = [row for row in dp_rows if int(row["solved_runs"]) == 0]
    max_time = max(float(row["mean_time_ms"]) for row in solved)
    timeout_y = max_time * 1.8

    fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=180)
    ax.plot(
        [int(row["n"]) * int(row["capacity"]) for row in solved],
        [float(row["mean_time_ms"]) for row in solved],
        marker="o",
        linewidth=2,
        color="#1f4e79",
        label="DP solved",
    )
    if timed_out:
        ax.scatter(
            [int(row["n"]) * int(row["capacity"]) for row in timed_out],
            [timeout_y for _ in timed_out],
            marker="x",
            s=70,
            color="#c62828",
            label="DP timeout",
        )
    ax.axvline(5_000_000, linestyle="--", linewidth=1.2, color="#555555", label="max-cells threshold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DP state space size (N x C)")
    ax.set_ylabel("Mean runtime (ms, log scale)")
    ax.set_title("Dynamic Programming Scalability")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeated synthetic knapsack performance experiments.")
    parser.add_argument("--output-dir", default="submission/results")
    parser.add_argument("--repetitions", type=int, default=100, help="Number of random instances per size.")
    parser.add_argument(
        "--max-cells",
        type=int,
        default=5_000_000,
        help="Skip exact DP when n*capacity exceeds this number.",
    )
    args = parser.parse_args()
    rows = run_experiments(Path(args.output_dir), max_cells=args.max_cells, repetitions=args.repetitions)
    print(f"Wrote {len(rows)} raw results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
