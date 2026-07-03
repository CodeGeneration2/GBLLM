from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import artifact_root, outputs_root


TOOL_NAME = "EffiSkill"
MODEL_ORDER = ["GPT-5-mini", "Qwen3-Coder-30B"]
LANGUAGE_ORDER = ["Python", "C++"]
METHOD_ORDER = ["Instruction", "RAG", "CoT", "SBLLM", "FasterPy", "EffiCoder", TOOL_NAME]
TABLE_METHOD_ORDER = ["Instruction", "RAG", "CoT", "SBLLM", "FasterPy", "EffiCoder", TOOL_NAME]
TOPK_VALUES = tuple(range(1, 9))
FIGURE_FILENAMES = {
    "topk": "figure4_topk_growth.pdf",
    "win_loss": "figure5_task_level_comparison.pdf",
    "bucket": "figure6_top8_bucket_distribution.pdf",
}


@dataclass(frozen=True)
class RunSpec:
    model: str
    method: str
    language: str
    path: str
    lang_key: str


RUN_SPECS = [
    RunSpec("GPT-5-mini", "Instruction", "Python", "gpt5-intstr", "python3"),
    RunSpec("GPT-5-mini", "RAG", "Python", "pie_rag_gpt5", "python3"),
    RunSpec("GPT-5-mini", "CoT", "Python", "gpt5mini_cot", "python3"),
    RunSpec("GPT-5-mini", "SBLLM", "Python", "sbllm_gpt", "python3"),
    RunSpec("GPT-5-mini", "FasterPy", "Python", "fasterpy_gpt5mini", "python3"),
    RunSpec("GPT-5-mini", TOOL_NAME, "Python", "gpt5_effiskill", "python3"),
    RunSpec("GPT-5-mini", "Instruction", "C++", "instru_gpt5_cpp", "cpp"),
    RunSpec("GPT-5-mini", "RAG", "C++", "pie_rag_gpt5_cpp", "cpp"),
    RunSpec("GPT-5-mini", "CoT", "C++", "gpt5_cot_cpp", "cpp"),
    RunSpec("GPT-5-mini", "SBLLM", "C++", "sbllm_cpp_gpt5", "cpp"),
    RunSpec("GPT-5-mini", "FasterPy", "C++", "fasterpy_gpt5min_cpp", "cpp"),
    RunSpec("GPT-5-mini", TOOL_NAME, "C++", "effiskill_cpp_gpt", "cpp"),
    RunSpec("Qwen3-Coder-30B", "Instruction", "Python", "qwen30b-instru", "python3"),
    RunSpec("Qwen3-Coder-30B", "RAG", "Python", "pie_rag_qwen", "python3"),
    RunSpec("Qwen3-Coder-30B", "CoT", "Python", "qwen_cot", "python3"),
    RunSpec("Qwen3-Coder-30B", "SBLLM", "Python", "sbllm_qwen", "python3"),
    RunSpec("Qwen3-Coder-30B", "FasterPy", "Python", "fasterpy_qwen", "python3"),
    RunSpec("Qwen3-Coder-30B", "EffiCoder", "Python", "efficoder_qwen", "python3"),
    RunSpec("Qwen3-Coder-30B", TOOL_NAME, "Python", "qwen30b-effiskill", "python3"),
    RunSpec("Qwen3-Coder-30B", "Instruction", "C++", "instru_qwen_cpp", "cpp"),
    RunSpec("Qwen3-Coder-30B", "RAG", "C++", "pie_rag_qwen_cpp", "cpp"),
    RunSpec("Qwen3-Coder-30B", "CoT", "C++", "qwen_cot_cpp", "cpp"),
    RunSpec("Qwen3-Coder-30B", "SBLLM", "C++", "sbllm_cpp_qwen", "cpp"),
    RunSpec("Qwen3-Coder-30B", "FasterPy", "C++", "fasterpy_qwen_cpp", "cpp"),
    RunSpec("Qwen3-Coder-30B", "EffiCoder", "C++", "efficoder_cpp", "cpp"),
    RunSpec("Qwen3-Coder-30B", TOOL_NAME, "C++", "effiskill_cpp_qwen", "cpp"),
]


def rq1_eval_dir() -> Path:
    return artifact_root() / "data" / "rq1" / "evaluation"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 18.0,
            "axes.labelsize": 17.0,
            "xtick.labelsize": 17.0,
            "ytick.labelsize": 17.0,
            "legend.fontsize": 17.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def lang_entry(entry: dict | None, lang_key: str) -> dict:
    if not isinstance(entry, dict):
        return {}
    return entry.get(lang_key) or {}


def runtime(entry: dict | None, lang_key: str) -> float | None:
    value = lang_entry(entry, lang_key).get("runtime_sum")
    return float(value) if value is not None else None


def passed(entry: dict | None, lang_key: str) -> bool:
    return bool(lang_entry(entry, lang_key).get("passed"))


def original_filename(lang_key: str) -> str:
    return "stats_canonical_cpp.json" if lang_key == "cpp" else "stats_original.json"


def collect_candidate_paths(split_dir: Path, lang_key: str) -> dict[int, Path]:
    orig_name = original_filename(lang_key)
    candidate_paths: list[tuple[int, Path]] = []
    for path in split_dir.glob("*.json"):
        if path.name == orig_name:
            continue
        for prefix in ("stats_rank_", "stats_direct_"):
            if path.name.startswith(prefix):
                candidate_paths.append((int(path.name[len(prefix) : -5]), path))
                break
    return dict(sorted(candidate_paths))


def improvement_ratio(original_runtime: float | None, candidate_runtime: float | None, is_correct: bool = True) -> float:
    if original_runtime is None or candidate_runtime is None or not is_correct:
        return 1.0
    return max(original_runtime / candidate_runtime, 1.0)


def problem_family(problem_id: str) -> str:
    return problem_id.split("_", 1)[0]


def load_method_task_frame(base_dir: Path, spec: RunSpec) -> pd.DataFrame:
    private_dir = base_dir / spec.path / "private"
    public_dir = base_dir / spec.path / "public"
    orig_name = original_filename(spec.lang_key)
    original_private = load_json(private_dir / orig_name)
    rank_private = {index: load_json(path) for index, path in collect_candidate_paths(private_dir, spec.lang_key).items()}
    rank_public = {index: load_json(path) for index, path in collect_candidate_paths(public_dir, spec.lang_key).items()}

    rows: list[dict] = []
    for problem_id in sorted(original_private):
        original_runtime = runtime(original_private.get(problem_id), spec.lang_key)
        row = {
            "problem_id": problem_id,
            "family": problem_family(problem_id),
            "Model": spec.model,
            "Language": spec.language,
            "Method": spec.method,
            "original_runtime": original_runtime,
        }

        public_valid: list[tuple[int, float]] = []
        for rank_index in sorted(rank_public):
            public_entry = rank_public[rank_index].get(problem_id)
            private_entry = rank_private[rank_index].get(problem_id)
            public_pass = passed(public_entry, spec.lang_key)
            private_pass = passed(private_entry, spec.lang_key)
            public_runtime = runtime(public_entry, spec.lang_key)
            private_runtime = runtime(private_entry, spec.lang_key)

            row[f"rank_{rank_index}_public_pass"] = public_pass
            row[f"rank_{rank_index}_private_pass"] = private_pass
            row[f"rank_{rank_index}_public_runtime"] = public_runtime
            row[f"rank_{rank_index}_private_runtime"] = private_runtime
            row[f"rank_{rank_index}_private_ratio"] = improvement_ratio(original_runtime, private_runtime, private_pass)

            if public_pass and public_runtime is not None:
                public_valid.append((rank_index, public_runtime))

        public_valid.sort(key=lambda item: item[1])
        ranked_candidates = [rank_index for rank_index, _ in public_valid]
        row["public_rank_order"] = "|".join(f"rank_{rank_index}" for rank_index in ranked_candidates)

        for k in TOPK_VALUES:
            selected_ranks = ranked_candidates[:k]
            ratios = [row[f"rank_{rank_index}_private_ratio"] for rank_index in selected_ranks]
            row[f"top{k}_ratio"] = max(ratios, default=1.0)
            row[f"top{k}_opt"] = float(row[f"top{k}_ratio"] >= 1.10)

        rows.append(row)

    return pd.DataFrame(rows)


def build_master_task_frame(base_dir: Path) -> pd.DataFrame:
    frames = [load_method_task_frame(base_dir, spec) for spec in RUN_SPECS]
    master = pd.concat(frames, ignore_index=True)
    master["Model"] = pd.Categorical(master["Model"], MODEL_ORDER, ordered=True)
    master["Language"] = pd.Categorical(master["Language"], LANGUAGE_ORDER, ordered=True)
    master["Method"] = pd.Categorical(master["Method"], METHOD_ORDER, ordered=True)
    return master.sort_values(["Language", "Model", "Method", "problem_id"]).reset_index(drop=True)


def summarize_topk(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (language, model, method), group in master.groupby(["Language", "Model", "Method"], observed=True):
        for k in TOPK_VALUES:
            rows.append(
                {
                    "Language": language,
                    "Model": model,
                    "Method": method,
                    "k": k,
                    "OPT (%)": group[f"top{k}_opt"].mean() * 100.0,
                }
            )
    return pd.DataFrame(rows)


def summarize_buckets(master: pd.DataFrame, ratio_col: str = "top8_ratio") -> pd.DataFrame:
    bins = [0.999999, 1.000001, 1.10, 1.25, 1.50, np.inf]
    labels = ["No improvement", "Mild (1.00-1.10)", "Moderate (1.10-1.25)", "Strong (1.25-1.50)", "Very strong (>1.50)"]
    rows = []
    for (language, model, method), group in master.groupby(["Language", "Model", "Method"], observed=True):
        bucketed = pd.cut(group[ratio_col], bins=bins, labels=labels, right=False, include_lowest=True)
        counts = bucketed.value_counts(normalize=True).reindex(labels, fill_value=0.0)
        for label in labels:
            rows.append(
                {
                    "Language": language,
                    "Model": model,
                    "Method": method,
                    "Bucket": label,
                    "Tasks (%)": counts[label] * 100.0,
                }
            )
    return pd.DataFrame(rows)


def strongest_baseline_lookup(summary: pd.DataFrame, metric_col: str, language: str, model: str) -> str:
    subset = summary[
        (summary["Language"] == language)
        & (summary["Model"] == model)
        & (summary["Method"] != TOOL_NAME)
    ].copy()
    subset = subset.sort_values([metric_col, "Method"], ascending=[False, True])
    return str(subset.iloc[0]["Method"])


def pairwise_task_compare(master: pd.DataFrame, metric_col: str = "top8_ratio") -> pd.DataFrame:
    summary = (
        master.groupby(["Language", "Model", "Method"], observed=True)[metric_col]
        .mean()
        .reset_index()
        .rename(columns={metric_col: "mean_metric"})
    )
    rows = []
    for language in LANGUAGE_ORDER:
        for model in MODEL_ORDER:
            baseline = strongest_baseline_lookup(summary, "mean_metric", language, model)
            tool = master[
                (master["Language"] == language) & (master["Model"] == model) & (master["Method"] == TOOL_NAME)
            ][["problem_id", metric_col]]
            base = master[
                (master["Language"] == language) & (master["Model"] == model) & (master["Method"] == baseline)
            ][["problem_id", metric_col]]
            merged = tool.merge(base, on="problem_id", suffixes=("_tool", "_base"))
            rows.append(
                {
                    "Language": language,
                    "Model": model,
                    "Baseline": baseline,
                    "EffiSkill Wins": int((merged[f"{metric_col}_tool"] > merged[f"{metric_col}_base"]).sum()),
                    "Ties": int(np.isclose(merged[f"{metric_col}_tool"], merged[f"{metric_col}_base"]).sum()),
                    "Losses": int((merged[f"{metric_col}_tool"] < merged[f"{metric_col}_base"]).sum()),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_one_sided_pvalue(delta: np.ndarray, reps: int = 10000, seed: int = 0, chunk_size: int = 2000) -> float:
    delta = np.asarray(delta, dtype=float)
    if len(delta) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    le_zero = 0
    remaining = reps
    while remaining > 0:
        batch = min(chunk_size, remaining)
        sample_index = rng.integers(0, len(delta), size=(batch, len(delta)))
        sample_means = delta[sample_index].mean(axis=1)
        le_zero += int(np.count_nonzero(sample_means <= 0))
        remaining -= batch
    return float((le_zero + 1) / (reps + 1))


def compute_opt_summary(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, str, str], str]]:
    rows = []
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    for spec in RUN_SPECS:
        frame = master[
            (master["Model"] == spec.model)
            & (master["Method"] == spec.method)
            & (master["Language"] == spec.language)
        ].copy()
        frames[(spec.model, spec.method, spec.language)] = frame
        rows.append(
            {
                "Model": spec.model,
                "Method": spec.method,
                "Language": spec.language,
                "OPT@1 (%)": frame["top1_opt"].mean() * 100.0,
                "OPT@8 (%)": frame["top8_opt"].mean() * 100.0,
            }
        )

    summary = pd.DataFrame(rows)
    summary["Model"] = pd.Categorical(summary["Model"], MODEL_ORDER, ordered=True)
    summary["Language"] = pd.Categorical(summary["Language"], LANGUAGE_ORDER, ordered=True)
    summary["Method"] = pd.Categorical(summary["Method"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["Model", "Language", "Method"]).reset_index(drop=True)

    marker_lookup: dict[tuple[str, str, str, str], str] = {}
    metrics = [("OPT@1 (%)", "top1_opt"), ("OPT@8 (%)", "top8_opt")]
    for model in MODEL_ORDER:
        for language in LANGUAGE_ORDER:
            group = summary[(summary["Model"] == model) & (summary["Language"] == language)]
            if group.empty:
                continue
            non_tool = group[group["Method"] != TOOL_NAME].copy()
            for metric_label, frame_metric in metrics:
                best_non_tool_value = float(non_tool[metric_label].max())
                tied = non_tool[non_tool[metric_label] == best_non_tool_value].sort_values("Method")
                best_non_tool_method = str(tied.iloc[0]["Method"])
                tool_frame = frames[(model, TOOL_NAME, language)].set_index("problem_id")
                baseline_frame = frames[(model, best_non_tool_method, language)].set_index("problem_id")
                aligned = tool_frame.join(baseline_frame, lsuffix="_tool", rsuffix="_base", how="inner")
                delta = aligned[f"{frame_metric}_tool"].to_numpy() - aligned[f"{frame_metric}_base"].to_numpy()
                seed = abs(hash((model, language, metric_label))) % (2**32)
                p_value = bootstrap_one_sided_pvalue(delta, seed=seed)
                tool_value = float(group[group["Method"] == TOOL_NAME][metric_label].iloc[0])
                if tool_value > best_non_tool_value and p_value < 0.05:
                    marker_lookup[(model, language, TOOL_NAME, metric_label)] = "*"
                elif tool_value > best_non_tool_value and p_value < 0.10:
                    marker_lookup[(model, language, TOOL_NAME, metric_label)] = "dagger"

    return summary, marker_lookup


def method_palette() -> dict[str, str]:
    return {
        "Instruction": "#7F8C8D",
        "RAG": "#D35400",
        "CoT": "#148F77",
        "SBLLM": "#8E44AD",
        "FasterPy": "#CA6F1E",
        "EffiCoder": "#5D6D7E",
        TOOL_NAME: "#C0392B",
    }


def iter_method_order(methods: list[str]) -> list[str]:
    method_set = {str(method) for method in methods}
    return [method for method in METHOD_ORDER if method in method_set]


def save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_topk_growth_pdf(topk_summary: pd.DataFrame, out_dir: Path) -> Path:
    palette = method_palette()
    fig, axes = plt.subplots(2, 2, figsize=(7.9, 6.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.945, bottom=0.26, wspace=0.30, hspace=0.48)
    settings = [(language, model) for language in LANGUAGE_ORDER for model in MODEL_ORDER]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (language, model), panel_label in zip(axes.ravel(), settings, panel_labels):
        subset = topk_summary[(topk_summary["Language"] == language) & (topk_summary["Model"] == model)]
        for method in iter_method_order(subset["Method"].tolist()):
            method_frame = subset[subset["Method"] == method].sort_values("k")
            color = palette[method]
            line_width = 2.8 if method == TOOL_NAME else 2.1
            marker_size = 5.8 if method == TOOL_NAME else 4.4
            alpha = 1.0 if method == TOOL_NAME else 0.78
            zorder = 4 if method == TOOL_NAME else 2
            ax.plot(
                method_frame["k"],
                method_frame["OPT (%)"],
                marker="o",
                markersize=marker_size,
                linewidth=line_width,
                color=color,
                alpha=alpha,
                zorder=zorder,
                label=method,
            )
        model_short = model.replace("Qwen3-Coder-30B-A3B-Instruct", "Qwen-30B")
        ax.set_title(f"{panel_label} {language} / {model_short}", loc="left", weight="bold", pad=4)
        ax.set_xlim(1, 8)
        ax.set_xticks(range(1, 9))
        ax.set_ylim(0, 75)
        ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("OPT@k (%)")
    fig.supxlabel("Candidate Budget $k$", y=0.115, fontsize=14.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005), columnspacing=1.6, handlelength=2.5)
    path = out_dir / FIGURE_FILENAMES["topk"]
    save_pdf(fig, path)
    return path


def plot_win_loss_pdf(win_loss: pd.DataFrame, out_dir: Path) -> Path:
    colors = {"EffiSkill Wins": "#C0392B", "Ties": "#B2BABB", "Losses": "#2471A3"}
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.8))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.935, bottom=0.24, wspace=0.30, hspace=0.62)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (_, row), panel_label in zip(axes.ravel(), win_loss.iterrows(), panel_labels):
        labels = ["EffiSkill Wins", "Ties", "Losses"]
        counts = np.array([row[label] for label in labels], dtype=float)
        total = counts.sum()
        widths = counts / total * 100.0
        left = 0.0
        for label, width, count in zip(labels, widths, counts):
            ax.barh([0], [width], left=left, color=colors[label], height=0.55)
            if width >= 9:
                ax.text(left + width / 2, 0, f"{int(count)}", ha="center", va="center", color="white", fontsize=12.2, weight="bold")
            else:
                ax.text(left + width + 1.2, 0, f"{int(count)}", ha="left", va="center", color=colors[label], fontsize=11.2, weight="bold")
            left += width
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([])
        ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.35)
        model_short = row["Model"].replace("Qwen3-Coder-30B-A3B-Instruct", "Qwen-30B")
        baseline_short = str(row["Baseline"]).replace("Qwen3-Coder-30B-A3B-Instruct", "Qwen-30B")
        ax.set_title(f"{panel_label} {row['Language']} / {model_short}\nvs {baseline_short}", loc="left", weight="bold", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supxlabel("Tasks (%)", y=0.12, fontsize=14.5)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in colors]
    fig.legend(legend_handles, list(colors), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    path = out_dir / FIGURE_FILENAMES["win_loss"]
    save_pdf(fig, path)
    return path


def plot_bucket_pdf(bucket_summary: pd.DataFrame, out_dir: Path) -> Path:
    bucket_order = [
        "No improvement",
        "Mild (1.00-1.10)",
        "Moderate (1.10-1.25)",
        "Strong (1.25-1.50)",
        "Very strong (>1.50)",
    ]
    bucket_colors = {
        "No improvement": "#D5D8DC",
        "Mild (1.00-1.10)": "#F7C86B",
        "Moderate (1.10-1.25)": "#76C893",
        "Strong (1.25-1.50)": "#4EA8DE",
        "Very strong (>1.50)": "#9D4EDD",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.05), sharex=True)
    fig.subplots_adjust(left=0.16, right=0.995, top=0.945, bottom=0.26, wspace=0.50, hspace=0.50)
    settings = [(language, model) for language in LANGUAGE_ORDER for model in MODEL_ORDER]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for ax, (language, model), panel_label in zip(axes.ravel(), settings, panel_labels):
        subset = bucket_summary[(bucket_summary["Language"] == language) & (bucket_summary["Model"] == model)]
        methods = iter_method_order(subset["Method"].tolist())
        y_positions = np.arange(len(methods))
        left = np.zeros(len(methods))
        for bucket in bucket_order:
            values = []
            for method in methods:
                row = subset[(subset["Method"] == method) & (subset["Bucket"] == bucket)]
                values.append(float(row["Tasks (%)"].iloc[0]) if not row.empty else 0.0)
            values = np.array(values)
            ax.barh(y_positions, values, left=left, color=bucket_colors[bucket], edgecolor="white", linewidth=0.7, label=bucket)
            left += values

        ax.set_yticks(y_positions)
        ax.set_yticklabels(methods)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.35)
        model_short = model.replace("Qwen3-Coder-30B-A3B-Instruct", "Qwen-30B")
        ax.set_title(f"{panel_label} {language} / {model_short}", loc="left", weight="bold", pad=4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=11.8, pad=1)
        for label in ax.get_yticklabels():
            if label.get_text() == TOOL_NAME:
                label.set_fontweight("bold")
                label.set_color("#C0392B")

    fig.supxlabel("Tasks (%)", y=0.12, fontsize=14.5)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=bucket_colors[bucket]) for bucket in bucket_order]
    fig.legend(legend_handles, bucket_order, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.005))
    path = out_dir / FIGURE_FILENAMES["bucket"]
    save_pdf(fig, path)
    return path


def format_display_value(value: float, marker: str, tex: bool = False) -> str:
    rounded = f"{value:.2f}"
    if marker == "*":
        return rounded + ("$^{*}$" if tex else "*")
    if marker == "dagger":
        return rounded + (r"$^{\dagger}$" if tex else "dagger")
    return rounded


def write_table_outputs(summary: pd.DataFrame, marker_lookup: dict[tuple[str, str, str, str], str], out_dir: Path) -> None:
    rows = []
    for model in MODEL_ORDER:
        for method in TABLE_METHOD_ORDER:
            method_rows = summary[(summary["Model"] == model) & (summary["Method"] == method)]
            if method_rows.empty:
                continue
            record = {"Model": model, "Method": method}
            for language in LANGUAGE_ORDER:
                lang_row = method_rows[method_rows["Language"] == language]
                if lang_row.empty:
                    continue
                for metric in ("OPT@1 (%)", "OPT@8 (%)"):
                    value = float(lang_row.iloc[0][metric])
                    marker = marker_lookup.get((model, language, method, metric), "")
                    col = f"{language} {metric}"
                    record[col] = format_display_value(value, marker, tex=False)
            rows.append(record)

    table_csv = pd.DataFrame(rows)
    table_csv.to_csv(out_dir / "table1_rq1.csv", index=False)

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Method & Python OPT@1 (\%) & Python OPT@8 (\%) & C++ OPT@1 (\%) & C++ OPT@8 (\%) \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        model_rows = summary[summary["Model"] == model]
        methods = [m for m in TABLE_METHOD_ORDER if m in set(model_rows["Method"].astype(str))]
        for idx, method in enumerate(methods):
            method_rows = model_rows[model_rows["Method"] == method]
            row = [model if idx == 0 else "", method]
            for language in LANGUAGE_ORDER:
                lang_row = method_rows[method_rows["Language"] == language]
                for metric in ("OPT@1 (%)", "OPT@8 (%)"):
                    value = float(lang_row.iloc[0][metric])
                    marker = marker_lookup.get((model, language, method, metric), "")
                    row.append(format_display_value(value, marker, tex=True))
            lines.append(" & ".join(row) + r" \\")
        if model != MODEL_ORDER[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out_dir / "table1_rq1.tex").write_text("\n".join(lines) + "\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the paper-facing RQ1 outputs.")
    parser.add_argument("--output-dir", type=Path, default=outputs_root() / "rq1")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    return parser


def run(output_dir: Path) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = rq1_eval_dir()
    master = build_master_task_frame(eval_dir)
    summary, marker_lookup = compute_opt_summary(master)
    topk_summary = summarize_topk(master)
    bucket_summary = summarize_buckets(master, ratio_col="top8_ratio")
    win_loss = pairwise_task_compare(master, metric_col="top8_ratio")
    write_table_outputs(summary, marker_lookup, output_dir)
    plot_topk_growth_pdf(topk_summary, output_dir)
    plot_win_loss_pdf(win_loss, output_dir)
    plot_bucket_pdf(bucket_summary, output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run(args.output_dir)


if __name__ == "__main__":
    main()
