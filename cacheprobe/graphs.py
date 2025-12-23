import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    """
    Configures matplotlib with specific styling.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def format_name(name: str) -> str:
    """
    Formats provider/scenario names for display in charts.

    :param name: Raw name with underscores (e.g., "openai_direct_same_account")
    :return: Human-readable formatted name with proper capitalization
    """
    result = name.replace("_", " ").title()
    result = result.replace("Openai", "OpenAI")
    result = result.replace("Byok", "BYOK")
    result = result.replace("Openrouter", "OpenRouter")
    return result


def generate_timing_histogram(result: dict, output_path: Path) -> Path:
    """
    Generates a histogram comparing cache hit vs miss timing distributions.

    Creates a dual-histogram overlay showing the distribution of Time To First Token
    for both cache hit and cache miss scenarios, with median lines and statistical
    annotations. Outliers beyond 1.5×IQR are excluded from visualization.

    :param result: Audit result dict containing hit_times, miss_times, and configuration
    :param output_path: Directory path to save the generated figure
    :return: Path to the generated histogram file
    """
    setup_style()

    hit_times = np.array(result["hit_times"])
    miss_times = np.array(result["miss_times"])
    config = result["configuration"]

    fig, ax = plt.subplots(figsize=(6, 4))

    all_times = np.concatenate([hit_times, miss_times])
    q1, q3 = np.percentile(all_times, [25, 75])
    iqr = q3 - q1
    x_min = max(0, q1 - 1.5 * iqr)
    x_max = q3 + 1.5 * iqr

    miss_filtered = miss_times[(miss_times >= x_min) & (miss_times <= x_max)]
    hit_filtered = hit_times[(hit_times >= x_min) & (hit_times <= x_max)]

    bins = np.linspace(x_min, x_max, 40)

    ax.hist(
        miss_filtered,
        bins=bins,
        alpha=0.6,
        label=f"Miss (n={len(miss_times)})",
        color="#2196F3",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.hist(
        hit_filtered,
        bins=bins,
        alpha=0.6,
        label=f"Hit (n={len(hit_times)})",
        color="#F44336",
        edgecolor="white",
        linewidth=0.5,
    )

    miss_median = np.median(miss_times)
    hit_median = np.median(hit_times)

    ax.axvline(
        miss_median,
        color="#1565C0",
        linestyle="--",
        linewidth=1.5,
        label=f"Miss median: {miss_median:.3f}s",
    )
    ax.axvline(
        hit_median,
        color="#C62828",
        linestyle="--",
        linewidth=1.5,
        label=f"Hit median: {hit_median:.3f}s",
    )

    ax.set_xlim(x_min, x_max * 1.02)
    ax.set_xlabel("Time to First Token (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        f"{format_name(config['provider'])} - {format_name(config['scenario'])}"
    )
    ax.legend(loc="upper right", framealpha=0.9)

    p_value = result["p_value"]
    ks_stat = result["ks_statistic"]
    cache_detected = result["cache_detected_by_statistical_test"]

    outliers_excluded = len(all_times) - len(miss_filtered) - len(hit_filtered)

    text = f"KS stat: {ks_stat:.3f}\np-value: {p_value:.2e}\nCache detected: {cache_detected}"
    if outliers_excluded > 0:
        text += f"\n({outliers_excluded} outliers not shown,\n>1.5×IQR from quartiles)"

    ax.text(
        0.98,
        0.75,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    provider = config["provider"]
    scenario = config["scenario"]
    filename = f"histogram_{provider}_{scenario}.pdf"
    filepath = output_path / filename

    plt.savefig(filepath, format="pdf")
    plt.close()

    return filepath


def generate_boxplot(result: dict, output_path: Path) -> Path:
    """
    Generates a boxplot comparing cache hit vs miss timing distributions.

    Creates side-by-side boxplots showing the quartile distribution of TTFT
    for hit and miss scenarios. Outliers beyond 1.5×IQR are excluded.

    :param result: Audit result dict containing hit_times, miss_times, and configuration
    :param output_path: Directory path to save the generated figure
    :return: Path to the generated boxplot file
    """
    setup_style()

    hit_times = np.array(result["hit_times"])
    miss_times = np.array(result["miss_times"])
    config = result["configuration"]

    fig, ax = plt.subplots(figsize=(4, 5))

    data = [miss_times, hit_times]
    bp = ax.boxplot(
        data,
        tick_labels=["Miss", "Hit"],
        patch_artist=True,
        widths=0.6,
        showfliers=False,
    )

    colors = ["#2196F3", "#F44336"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    def get_whisker_bounds(times):
        q1, q3 = np.percentile(times, [25, 75])
        iqr = q3 - q1
        w_low = q1 - 1.5 * iqr
        w_high = q3 + 1.5 * iqr
        actual_low = (
            times[times >= w_low].min() if np.any(times >= w_low) else times.min()
        )
        actual_high = (
            times[times <= w_high].max() if np.any(times <= w_high) else times.max()
        )
        return actual_low, actual_high

    miss_low, miss_high = get_whisker_bounds(miss_times)
    hit_low, hit_high = get_whisker_bounds(hit_times)

    whisker_low = min(miss_low, hit_low)
    whisker_high = max(miss_high, hit_high)

    y_range = whisker_high - whisker_low
    y_min = max(0, whisker_low - 0.1 * y_range)
    y_max = whisker_high + 0.1 * y_range
    ax.set_ylim(y_min, y_max)

    all_times = np.concatenate([hit_times, miss_times])
    outlier_count = np.sum((all_times < whisker_low) | (all_times > whisker_high))

    if outlier_count > 0:
        ax.text(
            0.98,
            0.98,
            f"{outlier_count} outliers not shown\n(>1.5×IQR from quartiles)",
            transform=ax.transAxes,
            fontsize=7,
            verticalalignment="top",
            horizontalalignment="right",
            style="italic",
            alpha=0.7,
        )

    ax.set_ylabel("Time to First Token (seconds)")
    ax.set_title(
        f"{format_name(config['provider'])} - {format_name(config['scenario'])}"
    )

    plt.tight_layout()

    provider = config["provider"]
    scenario = config["scenario"]
    filename = f"boxplot_{provider}_{scenario}.pdf"
    filepath = output_path / filename

    plt.savefig(filepath, format="pdf")
    plt.close()

    return filepath


def generate_comparison_chart(results: list[dict], output_path: Path) -> Path:
    """
    Generates a grouped bar chart comparing timing across multiple scenarios.

    Creates a chart with grouped bars showing mean TTFT with standard deviation
    error bars for both hit and miss conditions across all provided results.

    :param results: List of audit result dicts to compare
    :param output_path: Directory path to save the generated figure
    :return: Path to the generated comparison chart file
    """
    setup_style()

    labels = []
    miss_means = []
    hit_means = []
    miss_stds = []
    hit_stds = []

    for r in results:
        config = r["configuration"]
        provider = config["provider"]
        scenario = config["scenario"]
        short_scenario = scenario.replace("_account", "").replace("_", "\n")
        labels.append(f"{provider}\n{short_scenario}")

        metrics = r["metrics"]
        miss_means.append(metrics["mean_miss_time"])
        hit_means.append(metrics["mean_hit_time"])
        miss_stds.append(metrics["std_miss_time"])
        hit_stds.append(metrics["std_hit_time"])

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.5), 5))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(
        x - width / 2,
        miss_means,
        width,
        label="Miss",
        color="#2196F3",
        alpha=0.8,
        yerr=miss_stds,
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        hit_means,
        width,
        label="Hit",
        color="#F44336",
        alpha=0.8,
        yerr=hit_stds,
        capsize=3,
    )

    ax.set_ylabel("Mean TTFT (seconds)")
    ax.set_title("Cache Timing Comparison Across Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()

    plt.tight_layout()

    filename = "comparison_chart.pdf"
    filepath = output_path / filename

    plt.savefig(filepath, format="pdf")
    plt.close()

    return filepath


def generate_metadata_chart(results: list[dict], output_path: Path) -> Path:
    """
    Generates a chart showing cache metadata disclosure rates by scenario.

    Creates a grouped bar chart showing the percentage of API responses that
    disclosed cache usage in their metadata for both hit and miss scenarios.

    :param results: List of audit result dicts to analyze
    :param output_path: Directory path to save the generated figure
    :return: Path to the generated metadata disclosure chart file
    """
    setup_style()

    labels = []
    hit_cache_pcts = []
    miss_cache_pcts = []
    has_data = []

    for r in results:
        config = r["configuration"]
        cache_analysis = r.get("cache_token_analysis", {})

        provider = config["provider"]
        scenario = config["scenario"]
        short_scenario = scenario.replace("_account", "").replace("_", "\n")
        labels.append(f"{provider}\n{short_scenario}")

        if cache_analysis.get("has_cache_data", False):
            hit_cache_pcts.append(cache_analysis.get("hit_cache_percentage", 0))
            miss_cache_pcts.append(cache_analysis.get("miss_cache_percentage", 0))
            has_data.append(True)
        else:
            hit_cache_pcts.append(0)
            miss_cache_pcts.append(0)
            has_data.append(False)

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.5), 5))

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(
        x - width / 2,
        hit_cache_pcts,
        width,
        label="Hit samples with cache metadata",
        color="#4CAF50",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        miss_cache_pcts,
        width,
        label="Miss samples with cache metadata",
        color="#FF9800",
        alpha=0.8,
    )

    for i, has in enumerate(has_data):
        if not has:
            ax.text(
                x[i],
                5,
                "No\nmetadata",
                ha="center",
                va="bottom",
                fontsize=7,
                style="italic",
                alpha=0.6,
            )

    ax.set_ylabel("Cache Disclosure Rate (%)")
    ax.set_title("API Metadata Cache Disclosure by Scenario")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)

    ax.axhline(y=50, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(len(labels) - 0.5, 52, "Detection threshold", fontsize=7, alpha=0.6)

    plt.tight_layout()

    filename = "metadata_disclosure_chart.pdf"
    filepath = output_path / filename

    plt.savefig(filepath, format="pdf")
    plt.close()

    return filepath


def main():
    """
    CLI entry point for generating graphs from CacheProbe results.

    Supports three modes:
    - Single file: Generate histogram and boxplot for one result file
    - Compare: Generate comparison chart for multiple result files
    - All: Process all files in results/ directory and generate all chart types
    """
    parser = argparse.ArgumentParser(
        description="Generate graphs from CacheProbe results for paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Single result file to generate graphs for",
    )
    parser.add_argument(
        "--compare",
        "-c",
        nargs="+",
        type=Path,
        help="Multiple result files to compare",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for generated files (default: paper/figures)",
    )

    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Generate all graph types for all files in results/",
    )

    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    generated_files = []

    if args.file:
        with open(args.file) as f:
            result = json.load(f)
        print(f"Generating graphs for: {args.file.name}")

        hist_path = generate_timing_histogram(result, args.output)
        print(f"  Created: {hist_path}")
        generated_files.append(hist_path)

        box_path = generate_boxplot(result, args.output)
        print(f"  Created: {box_path}")
        generated_files.append(box_path)

    if args.compare:
        results = [json.load(open(fp)) for fp in args.compare]
        print(f"Generating comparison chart for {len(results)} files")

        chart_path = generate_comparison_chart(results, args.output)
        print(f"  Created: {chart_path}")
        generated_files.append(chart_path)

    if args.all:
        results_dir = Path("results")
        if not results_dir.exists():
            print("Error: results/ directory not found")
            exit(1)

        result_files = sorted(results_dir.glob("*.json"))
        print(f"Processing {len(result_files)} result files...")

        for rf in result_files:
            with open(rf) as f:
                result = json.load(f)
            print(f"\nGenerating graphs for: {rf.name}")

            hist_path = generate_timing_histogram(result, args.output)
            print(f"  Created: {hist_path}")
            generated_files.append(hist_path)

            box_path = generate_boxplot(result, args.output)
            print(f"  Created: {box_path}")
            generated_files.append(box_path)

        results = [json.load(open(rf)) for rf in result_files]

        chart_path = generate_comparison_chart(results, args.output)
        print(f"\nCreated comparison chart: {chart_path}")
        generated_files.append(chart_path)

        metadata_path = generate_metadata_chart(results, args.output)
        print(f"Created metadata disclosure chart: {metadata_path}")
        generated_files.append(metadata_path)

    if not generated_files:
        parser.print_help()
        exit(1)

    print(f"\nGenerated {len(generated_files)} files in {args.output}")


if __name__ == "__main__":
    main()
