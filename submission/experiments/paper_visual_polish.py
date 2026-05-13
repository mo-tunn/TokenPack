from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "submission" / "paper"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"
COMPRESSION_CSV = ROOT / "submission" / "results" / "qasper_compression_report_hybrid_greedy" / "qasper_compression_methods.csv"
LONGBENCH_CSV = ROOT / "submission" / "results" / "longbench_v2_modal_hybrid_greedy_83_latency" / "longbench_generation_summary.csv"
SCALING_64K_CSV = ROOT / "submission" / "results" / "longbench_v2_modal_64k_diagnostic_selectors" / "longbench_generation_summary.csv"
SCALING_128K_CSV = ROOT / "submission" / "results" / "longbench_v2_modal_128k_diagnostic_selectors" / "longbench_generation_summary.csv"

NAVY = "#1d3557"
BLUE = "#3a7ca5"
LIGHT_BLUE = "#eaf4ff"
ORANGE = "#c96f2d"
GRAY = "#6b7280"
DARK = "#1f2937"


def main() -> int:
    polish_tables(TABLE_DIR)
    write_visual_summary(COMPRESSION_CSV, LONGBENCH_CSV, FIGURE_DIR / "tokenpack_visual_summary.png")
    write_scaling_diagnostic(SCALING_64K_CSV, SCALING_128K_CSV, FIGURE_DIR / "longbench_scaling_diagnostic.png")
    print(f"Polished tables under {TABLE_DIR}")
    print(f"Wrote {FIGURE_DIR / 'tokenpack_visual_summary.png'}")
    print(f"Wrote {FIGURE_DIR / 'longbench_scaling_diagnostic.png'}")
    return 0


def polish_tables(table_dir: Path) -> None:
    for path in sorted(table_dir.glob("*.tex")):
        text = path.read_text(encoding="utf-8")
        polished = _booktabs_table(text)
        path.write_text(polished, encoding="utf-8")


def _booktabs_table(text: str) -> str:
    lines = text.splitlines()
    hline_indexes = [index for index, line in enumerate(lines) if line.strip() == r"\hline"]
    if len(hline_indexes) >= 3:
        lines[hline_indexes[0]] = r"\toprule"
        lines[hline_indexes[1]] = r"\midrule"
        lines[hline_indexes[-1]] = r"\bottomrule"
        for index in hline_indexes[2:-1]:
            lines[index] = r"\midrule"

    for index, line in enumerate(lines):
        if line.strip().startswith(r"\begin{tabular}") and "@{}" not in line:
            lines[index] = _compact_tabular(line)
            break

    for index, line in enumerate(lines):
        if line.strip() == r"\toprule":
            header_index = index + 1
            if header_index < len(lines) and not lines[header_index].strip().startswith(r"\rowcolor{tablehead}"):
                lines.insert(header_index, r"\rowcolor{tablehead}")
            break

    lines = [_highlight_tokenpack_row(line) for line in lines]
    return "\n".join(lines) + "\n"


def _compact_tabular(line: str) -> str:
    prefix = r"\begin{tabular}{"
    suffix = "}"
    stripped = line.strip()
    if not stripped.startswith(prefix) or not stripped.endswith(suffix):
        return line
    spec = stripped[len(prefix) : -len(suffix)]
    compact = f"{prefix}@{{}}{spec}@{{}}{suffix}"
    return line.replace(stripped, compact)


def _highlight_tokenpack_row(line: str) -> str:
    stripped = line.strip()
    if not stripped.endswith(r"\\") or stripped.startswith((r"\rowcolor", r"\toprule", r"\midrule", r"\bottomrule")):
        return line
    tokens = ("Only TokenPack", "TokenPack +", "TP-HG", "budget-top-k")
    if not any(token in line for token in tokens):
        return line
    leading = line[: len(line) - len(line.lstrip())]
    return f"{leading}\\rowcolor{{tokenpackhighlight}}{line.lstrip()}"


def write_visual_summary(compression_csv: Path, longbench_csv: Path, output_path: Path) -> None:
    compression_rows = _load_csv_rows(compression_csv)
    longbench_rows = _load_csv_rows(longbench_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.45),
        dpi=220,
        gridspec_kw={"width_ratios": [1.15, 0.92, 0.92]},
    )
    _draw_longbench_operating_points(axes[0], longbench_rows)
    _draw_qasper_evidence_bars(axes[1], compression_rows)
    _draw_latency_bars(axes[2], longbench_rows)
    fig.suptitle("TokenPack result dashboard", x=0.5, y=0.995, fontsize=10.5, weight="bold", color=NAVY)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.80, bottom=0.22, wspace=0.42)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _method_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    return next(row for row in rows if row["method"] == method)


def _draw_longbench_operating_points(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    method_order = [
        ("Full", "full-context", "#9ca3af"),
        ("Prod-RAG", "production-rag-50", "#64748b"),
        ("TP-HG", "tokenpack-50", NAVY),
        ("LLL", "only-longllmlingua-rate050", ORANGE),
        ("TP+LLL", "tokenpack-50+longllmlingua-rate050", "#0f766e"),
    ]
    ax.set_title("LongBench: quality per context", color=NAVY, fontsize=8.7, weight="bold", pad=6)
    ax.grid(True, color="#e5e7eb", linewidth=0.75)
    ax.set_axisbelow(True)
    for label, method, color in method_order:
        row = _method_row(rows, method)
        if int(row["runs"]) == 0:
            continue
        x = float(row["avg_context_tokens"]) / 1000.0
        y = float(row["accuracy"])
        is_tp = label.startswith("TP")
        ax.scatter(
            [x],
            [y],
            s=58 if is_tp else 43,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        dx = -8 if label == "Full" else 4
        dy = 7 if label in {"TP-HG", "TP+LLL"} else -12
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=6.6, color=DARK, weight="bold" if is_tp else "normal")
    ax.annotate(
        "+15.6% rel. acc.\nvs full",
        xy=(8.73, 0.446),
        xytext=(10.3, 0.462),
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 0.9},
        fontsize=6.5,
        color=NAVY,
        weight="bold",
    )
    ax.set_xlabel("Avg. context tokens (k)", fontsize=7.0)
    ax.set_ylabel("MC accuracy", fontsize=7.0)
    ax.set_xlim(3.5, 18.8)
    ax.set_ylim(0.345, 0.47)
    ax.tick_params(axis="both", labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


def _draw_qasper_evidence_bars(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    selected = [
        ("LLM2\n50%", next(row for row in rows if row["method"].startswith("Only LLMLingua-2")), ORANGE),
        ("TP\n51%", next(row for row in rows if row["method"] == "Only TokenPack"), NAVY),
        ("TP+LLM2\n58%", next(row for row in rows if row["method"].startswith("TokenPack + LLMLingua-2 rate=0.85")), BLUE),
    ]
    labels = [item[0] for item in selected]
    values = [float(item[1]["evidence_recall"]) for item in selected]
    colors = [item[2] for item in selected]
    ax.set_title("QASPER: evidence kept", color=NAVY, fontsize=8.7, weight="bold", pad=6)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", va="bottom", fontsize=6.7, weight="bold", color=DARK)
    ax.annotate(
        "+31% rel.\nvs LLM2",
        xy=(1, 0.934),
        xytext=(1.32, 0.955),
        arrowprops={"arrowstyle": "->", "color": NAVY, "lw": 0.85},
        fontsize=6.4,
        color=NAVY,
        weight="bold",
    )
    ax.set_ylim(0.55, 1.0)
    ax.set_ylabel("Evidence recall", fontsize=7.0)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


def _draw_latency_bars(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    selected = [
        ("Full", _method_row(rows, "full-context"), "#9ca3af"),
        ("Prod", _method_row(rows, "production-rag-50"), "#64748b"),
        ("TP", _method_row(rows, "tokenpack-50"), NAVY),
        ("TP+LLL", _method_row(rows, "tokenpack-50+longllmlingua-rate050"), "#0f766e"),
    ]
    labels = [item[0] for item in selected]
    values = [float(item[1]["avg_total_latency_seconds"]) for item in selected]
    colors = [item[2] for item in selected]
    ax.set_title("Hot-model latency", color=NAVY, fontsize=8.7, weight="bold", pad=6)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    full = values[0]
    for bar, value in zip(bars, values, strict=True):
        label = f"{full / value:.1f}x" if value else ""
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.13, label, ha="center", va="bottom", fontsize=6.6, weight="bold", color=DARK)
    ax.set_ylim(0, 4.8)
    ax.set_ylabel("Mean seconds", fontsize=7.0)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.75)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


def write_scaling_diagnostic(csv_64k: Path, csv_128k: Path, output_path: Path) -> None:
    windows = [
        ("32k--58k\nn=30", _load_csv_rows(csv_64k)),
        ("64k--112k\nn=8", _load_csv_rows(csv_128k)),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (accuracy_ax, context_ax) = plt.subplots(
        1,
        2,
        figsize=(7.2, 2.35),
        dpi=220,
        gridspec_kw={"width_ratios": [1.02, 1.0]},
    )
    _draw_scaling_accuracy(accuracy_ax, windows)
    _draw_scaling_context(context_ax, windows)
    fig.suptitle("LongBench larger-context diagnostic", x=0.5, y=0.995, fontsize=10.5, weight="bold", color=NAVY)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.78, bottom=0.30, wspace=0.32)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _scaling_methods(include_dp: bool = True) -> list[tuple[str, str, str]]:
    methods = [
        ("Full", "full-context", "#9ca3af"),
        ("Prod-RAG", "production-rag-50", "#64748b"),
        ("TP-HG", "hybrid-greedy-50", NAVY),
    ]
    if include_dp:
        methods.append(("Hybrid DP", "hybrid-knapsack-50", "#0f766e"))
    return methods


def _draw_scaling_accuracy(ax: plt.Axes, windows: list[tuple[str, list[dict[str, str]]]]) -> None:
    methods = _scaling_methods(include_dp=True)
    width = 0.18
    centers = list(range(len(windows)))
    offsets = [(-1.5 + index) * width for index in range(len(methods))]
    for offset, (label, method, color) in zip(offsets, methods, strict=True):
        values = [float(_method_row(rows, method)["accuracy"]) for _, rows in windows]
        bars = ax.bar([center + offset for center in centers], values, width=width, label=label, color=color)
        for bar, (_, rows) in zip(bars, windows, strict=True):
            row = _method_row(rows, method)
            runs = int(row["runs"])
            correct = int(round(float(row["accuracy"]) * runs))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.014,
                f"{correct}/{runs}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=DARK,
                weight="bold" if label == "TP-HG" else "normal",
            )
    ax.set_title("Accuracy by source window", color=NAVY, fontsize=8.7, weight="bold", pad=6)
    ax.set_ylabel("MC accuracy", fontsize=7.0)
    ax.set_xticks(centers, [label for label, _ in windows])
    ax.set_ylim(0.24, 0.56)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=5.8, frameon=False, handlelength=1.1)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


def _draw_scaling_context(ax: plt.Axes, windows: list[tuple[str, list[dict[str, str]]]]) -> None:
    methods = _scaling_methods(include_dp=False)
    width = 0.23
    centers = list(range(len(windows)))
    offsets = [(-1 + index) * width for index in range(len(methods))]
    for offset, (label, method, color) in zip(offsets, methods, strict=True):
        values = [float(_method_row(rows, method)["avg_context_tokens"]) / 1000.0 for _, rows in windows]
        bars = ax.bar([center + offset for center in centers], values, width=width, label=label, color=color)
        for bar, (_, rows) in zip(bars, windows, strict=True):
            row = _method_row(rows, method)
            if method == "full-context":
                text = "1.0x"
            else:
                text = f"{float(row['speedup_vs_full']):.1f}x"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                text,
                ha="center",
                va="bottom",
                fontsize=6.1,
                color=DARK,
                weight="bold",
            )
    ax.set_title("Context size and latency speedup", color=NAVY, fontsize=8.7, weight="bold", pad=6)
    ax.set_ylabel("Avg. context tokens (k)", fontsize=7.0)
    ax.set_xticks(centers, [label for label, _ in windows])
    ax.set_ylim(0, 88)
    ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=5.8, frameon=False, handlelength=1.1)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


def _draw_pipeline(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Selection-first context pipeline", color=NAVY, fontsize=9, weight="bold", pad=8)

    labels = [
        "Doc",
        "Struct.\nchunks",
        "Evidence\nscore",
        "Hybrid\ngreedy",
        "Optional\nLLM2",
        "LLM\nanswer",
    ]
    xs = [0.075, 0.245, 0.415, 0.585, 0.755, 0.925]
    y = 0.56
    box_w = 0.112
    box_h = 0.28
    for index, (x, label) in enumerate(zip(xs, labels)):
        is_core = index in {2, 3}
        face = LIGHT_BLUE if is_core else "#f8fafc"
        edge = BLUE if is_core else "#9ca3af"
        patch = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.018,rounding_size=0.028",
            linewidth=1.1,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x, y, label, ha="center", va="center", fontsize=5.8, color=DARK, weight="bold" if is_core else "normal")
        if index < len(xs) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_w / 2 + 0.012, y),
                    (xs[index + 1] - box_w / 2 - 0.012, y),
                    arrowstyle="-|>",
                    mutation_scale=8,
                    linewidth=1.0,
                    color=GRAY,
                )
            )
    ax.text(
        0.5,
        0.16,
        "A source-relative token budget is enforced before optional prompt compression.",
        ha="center",
        va="center",
        fontsize=7.1,
        color=GRAY,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 1)


def _draw_frontier(ax: plt.Axes, rows: list[dict[str, str]]) -> None:
    ax.set_title("QASPER compression frontier", color=NAVY, fontsize=9, weight="bold", pad=8)
    ax.grid(True, axis="both", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)

    tokenpack_rows = [row for row in rows if row["method"].startswith("TokenPack + LLMLingua-2")]
    tokenpack_rows.sort(key=lambda row: float(row["token_saving"]))
    only_tp = next(row for row in rows if row["method"] == "Only TokenPack")
    only_llm = next(row for row in rows if row["method"].startswith("Only LLMLingua-2"))

    x_tp = [100.0 * float(row["token_saving"]) for row in tokenpack_rows]
    y_tp = [float(row["evidence_recall"]) for row in tokenpack_rows]
    ax.plot(x_tp, y_tp, color=BLUE, linewidth=1.8, marker="o", markersize=4.5, label="TP + LLMLingua-2")
    ax.scatter(
        [100.0 * float(only_tp["token_saving"])],
        [float(only_tp["evidence_recall"])],
        color=NAVY,
        marker="D",
        s=38,
        label="Only TokenPack",
        zorder=4,
    )
    ax.scatter(
        [100.0 * float(only_llm["token_saving"])],
        [float(only_llm["evidence_recall"])],
        color=ORANGE,
        marker="s",
        s=36,
        label="Only LLMLingua-2",
        zorder=4,
    )

    for row in tokenpack_rows:
        rate = float(row["compression_rate"])
        x = 100.0 * float(row["token_saving"])
        y = float(row["evidence_recall"])
        ax.annotate(f"{rate:.2f}", (x, y), textcoords="offset points", xytext=(4, -9), fontsize=6.4, color=GRAY)

    ax.set_xlabel("Token saving (%)", fontsize=7.3)
    ax.set_ylabel("Evidence recall", fontsize=7.3)
    ax.tick_params(axis="both", labelsize=6.8)
    ax.set_xlim(48, 78)
    ax.set_ylim(0.56, 0.96)
    ax.legend(loc="lower left", fontsize=6.2, frameon=True, framealpha=0.92, borderpad=0.4, handlelength=1.3)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


if __name__ == "__main__":
    raise SystemExit(main())
