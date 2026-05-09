from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "submission" / "paper"
TABLE_DIR = PAPER_DIR / "tables"
FIGURE_DIR = PAPER_DIR / "figures"
COMPRESSION_CSV = ROOT / "submission" / "results" / "qasper_compression_report" / "qasper_compression_methods.csv"

NAVY = "#1d3557"
BLUE = "#3a7ca5"
LIGHT_BLUE = "#eaf4ff"
ORANGE = "#c96f2d"
GRAY = "#6b7280"
DARK = "#1f2937"


def main() -> int:
    polish_tables(TABLE_DIR)
    write_visual_summary(COMPRESSION_CSV, FIGURE_DIR / "tokenpack_visual_summary.png")
    print(f"Polished tables under {TABLE_DIR}")
    print(f"Wrote {FIGURE_DIR / 'tokenpack_visual_summary.png'}")
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
    tokens = ("TokenPack", "TP-50", "knapsack-redundancy")
    if not any(token in line for token in tokens):
        return line
    leading = line[: len(line) - len(line.lstrip())]
    return f"{leading}\\rowcolor{{tokenpackhighlight}}{line.lstrip()}"


def write_visual_summary(csv_path: Path, output_path: Path) -> None:
    rows = _load_compression_rows(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (pipeline_ax, frontier_ax) = plt.subplots(
        1,
        2,
        figsize=(7.1, 2.25),
        dpi=220,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    _draw_pipeline(pipeline_ax)
    _draw_frontier(frontier_ax, rows)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.86, bottom=0.18, wspace=0.28)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _load_compression_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _draw_pipeline(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title("Selection-first context pipeline", color=NAVY, fontsize=9, weight="bold", pad=8)

    labels = [
        "Document",
        "Struct.-aware\nchunks",
        "Hybrid\nscoring",
        "Knapsack\nbudget",
        "Optional\ncompressor",
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
        ax.text(x, y, label, ha="center", va="center", fontsize=6.3, color=DARK, weight="bold" if is_core else "normal")
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
        "Token budget is enforced before optional prompt compression.",
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
    ax.set_ylim(0.56, 0.94)
    ax.legend(loc="lower left", fontsize=6.2, frameon=True, framealpha=0.92, borderpad=0.4, handlelength=1.3)
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.8)


if __name__ == "__main__":
    raise SystemExit(main())
