"""
Analyze consistency of LLM responses across paraphrased prompts.
Computes:
1. Answer Agreement Rate (AAR): fraction of paraphrases with matching answers
2. Pairwise Answer Agreement (PAA): pairwise comparison across all version pairs
3. Reasoning Semantic Similarity (RSS): TF-IDF cosine similarity of reasoning chains
4. Statistical tests across complexity tiers
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for no-display environments
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

WORKSPACE = Path("/workspaces/temporal_consistency_of_llm_re_20260306_185941_1cc91799")
RESULTS_DIR = WORKSPACE / "results" / "data"
PLOTS_DIR = WORKSPACE / "results" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["claude-haiku", "claude-sonnet"]
TIERS = ["simple", "moderate", "complex"]
TIER_LABELS = {
    "simple": "Simple\n(GSM8K)",
    "moderate": "Moderate\n(MATH L1-2)",
    "complex": "Complex\n(MATH L3-5)"
}


def normalize_answer(answer):
    """Normalize answer string for comparison."""
    if not answer:
        return ""
    # Remove LaTeX formatting
    answer = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'[\$\\,]', '', answer)
    # Remove leading/trailing brackets
    answer = re.sub(r'^[\[\(](.+)[\]\)]$', r'\1', answer.strip())
    # Remove spaces and convert to lowercase
    answer = re.sub(r'\s+', '', answer.lower())
    # Remove trailing punctuation
    answer = answer.rstrip('.')
    # Remove common prefixes
    for prefix in ['answer:', 'finalanswer:', 'theanswer']:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
    return answer.strip()


def compute_answer_agreement(answers):
    """
    Compute answer agreement rate.
    Returns: (agreement_rate, majority_answer)
    Agreement rate = fraction of answers matching the mode (majority answer).
    """
    if not answers:
        return 0.0, ""
    normalized = [normalize_answer(a) for a in answers if a]
    if not normalized:
        return 0.0, ""
    counter = Counter(normalized)
    majority, majority_count = counter.most_common(1)[0]
    agreement_rate = majority_count / len(normalized)
    return agreement_rate, majority


def compute_pairwise_agreement(answers):
    """Compute pairwise agreement: fraction of all pairs that agree."""
    normalized = [normalize_answer(a) for a in answers if a]
    if len(normalized) < 2:
        return 1.0
    n = len(normalized)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    agree_count = sum(1 for i, j in pairs if normalized[i] == normalized[j])
    return agree_count / len(pairs)


def compute_tfidf_similarity(texts):
    """
    Compute mean pairwise TF-IDF cosine similarity.
    Measures how similar the reasoning chains are lexically/structurally.
    """
    valid_texts = [t for t in texts if t and len(t.strip()) > 10]
    if len(valid_texts) < 2:
        return 1.0

    try:
        vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        n = tfidf_matrix.shape[0]
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sims = []
        for i, j in pairs:
            sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
            sims.append(float(sim))
        return float(np.mean(sims)) if sims else 1.0
    except Exception:
        return float('nan')


def compute_original_vs_paraphrase_agreement(answers):
    """Compare original (index 0) against each paraphrase."""
    if len(answers) < 2:
        return 1.0
    orig_norm = normalize_answer(answers[0])
    paraphrase_norms = [normalize_answer(a) for a in answers[1:] if a]
    if not paraphrase_norms:
        return 1.0
    matches = sum(1 for p in paraphrase_norms if p == orig_norm)
    return matches / len(paraphrase_norms)


def load_responses(model_name):
    """Load LLM responses for a given model."""
    path = RESULTS_DIR / f"responses_{model_name}.json"
    if not path.exists():
        print(f"Warning: {path} not found")
        return []
    with open(path) as f:
        return json.load(f)


def analyze_model(responses, model_name):
    """Analyze consistency for all problems from a given model."""
    records = []

    for entry in responses:
        problem_id = entry["problem_id"]
        tier = entry["tier"]
        version_responses = entry.get("version_responses", [])

        if not version_responses:
            continue

        # Only analyze problems with 3+ versions (meaningful consistency measure)
        if len(version_responses) < 3:
            continue

        # Extract answers and full responses
        answers = [v["extracted_answer"] for v in version_responses]
        full_responses = [v["full_response"] for v in version_responses]

        # Compute answer consistency metrics
        aar, majority = compute_answer_agreement(answers)
        paa = compute_pairwise_agreement(answers)
        orig_vs_para = compute_original_vs_paraphrase_agreement(answers)

        # Compute reasoning chain similarity
        rss = compute_tfidf_similarity(full_responses)

        records.append({
            "problem_id": problem_id,
            "tier": tier,
            "model": model_name,
            "answer_agreement_rate": aar,
            "pairwise_answer_agreement": paa,
            "original_vs_paraphrase_agreement": orig_vs_para,
            "reasoning_similarity": rss,
            "n_versions": len(answers),
            "majority_answer": majority,
            "all_answers": answers,
        })

    return pd.DataFrame(records)


def run_statistical_tests(df):
    """Run Kruskal-Wallis and Mann-Whitney U tests across tiers."""
    results = {}
    metrics = [
        "answer_agreement_rate",
        "pairwise_answer_agreement",
        "original_vs_paraphrase_agreement",
        "reasoning_similarity"
    ]

    for metric in metrics:
        groups = {}
        for tier in TIERS:
            tier_data = df[df["tier"] == tier][metric].dropna().values
            groups[tier] = tier_data

        valid_groups = [g for g in groups.values() if len(g) > 0]
        if len(valid_groups) < 2:
            continue

        # Kruskal-Wallis test (non-parametric ANOVA)
        try:
            stat, p_kw = stats.kruskal(*[groups[t] for t in TIERS if len(groups[t]) > 0])
        except Exception:
            stat, p_kw = float('nan'), float('nan')

        # Pairwise Mann-Whitney U tests
        pairwise = {}
        tier_pairs = [("simple", "moderate"), ("moderate", "complex"), ("simple", "complex")]
        for t1, t2 in tier_pairs:
            if len(groups[t1]) > 0 and len(groups[t2]) > 0:
                try:
                    u_stat, p_mw = stats.mannwhitneyu(
                        groups[t1], groups[t2], alternative='two-sided'
                    )
                    n1, n2 = len(groups[t1]), len(groups[t2])
                    effect_size = 1 - (2 * u_stat) / (n1 * n2)
                    pairwise[f"{t1}_vs_{t2}"] = {
                        "u_stat": float(u_stat),
                        "p_value": float(p_mw),
                        "effect_size_r": float(effect_size),
                    }
                except Exception:
                    pass

        # Spearman correlation between tier index and metric
        tier_order = {"simple": 0, "moderate": 1, "complex": 2}
        tier_indices = df["tier"].map(tier_order)
        metric_values = df[metric]
        mask = metric_values.notna() & tier_indices.notna()
        if mask.sum() > 2:
            try:
                spearman_r, spearman_p = stats.spearmanr(
                    tier_indices[mask], metric_values[mask]
                )
            except Exception:
                spearman_r, spearman_p = float('nan'), float('nan')
        else:
            spearman_r, spearman_p = float('nan'), float('nan')

        results[metric] = {
            "kruskal_wallis": {"stat": float(stat), "p_value": float(p_kw)},
            "pairwise": pairwise,
            "spearman": {"r": float(spearman_r), "p_value": float(spearman_p)},
            "means_by_tier": {
                t: float(np.mean(groups[t])) if len(groups[t]) > 0 else None
                for t in TIERS
            },
            "stds_by_tier": {
                t: float(np.std(groups[t])) if len(groups[t]) > 0 else None
                for t in TIERS
            },
            "n_by_tier": {t: len(groups[t]) for t in TIERS},
        }

    return results


def plot_consistency_by_tier(all_df):
    """Create box plots of consistency metrics by tier and model."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "LLM Reasoning Consistency Under Paraphrased Prompts\n"
        "Lower values = more inconsistency across paraphrases",
        fontsize=13, fontweight='bold'
    )

    metrics = [
        ("answer_agreement_rate", "Answer Agreement Rate", "Fraction of versions\nwith matching final answer"),
        ("pairwise_answer_agreement", "Pairwise Answer Agreement", "Fraction of all answer pairs\nthat agree"),
        ("original_vs_paraphrase_agreement", "Original vs Paraphrase", "Fraction of paraphrases\nmatching original answer"),
        ("reasoning_similarity", "Reasoning Chain Similarity", "TF-IDF cosine similarity\nof reasoning chains"),
    ]

    colors = {'claude-haiku': '#2196F3', 'claude-sonnet': '#FF9800'}
    tier_positions = {tier: i for i, tier in enumerate(TIERS)}

    for ax, (metric, title, ylabel) in zip(axes.flatten(), metrics):
        model_offsets = {'claude-haiku': -0.15, 'claude-sonnet': 0.15}

        for model_name in MODELS:
            model_df = all_df[all_df["model"] == model_name]
            if model_df.empty:
                continue

            positions = []
            data_groups = []
            for tier in TIERS:
                tier_data = model_df[model_df["tier"] == tier][metric].dropna().values
                if len(tier_data) > 0:
                    positions.append(tier_positions[tier] + model_offsets[model_name])
                    data_groups.append(tier_data)

            if positions and data_groups:
                bp = ax.boxplot(
                    data_groups,
                    positions=positions,
                    widths=0.25,
                    patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                )
                for patch in bp['boxes']:
                    patch.set_facecolor(colors.get(model_name, 'gray'))
                    patch.set_alpha(0.7)

        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(len(TIERS)))
        ax.set_xticklabels([TIER_LABELS[t] for t in TIERS], fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], alpha=0.7, label=m)
            for m in MODELS
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "consistency_by_tier.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/consistency_by_tier.png")


def plot_complexity_trend(all_df, stat_results):
    """Plot mean consistency vs complexity tier with error bars and significance markers."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Consistency Degradation Across Complexity Tiers", fontsize=13, fontweight='bold')

    metrics_to_plot = [
        ("answer_agreement_rate", "Answer Agreement Rate"),
        ("reasoning_similarity", "Reasoning Chain Similarity (TF-IDF)"),
    ]

    colors = {'claude-haiku': '#2196F3', 'claude-sonnet': '#FF9800'}
    tier_x = [0, 1, 2]

    for ax, (metric, title) in zip(axes, metrics_to_plot):
        for model_name in MODELS:
            model_df = all_df[all_df["model"] == model_name]
            if model_df.empty:
                continue

            means = []
            sems = []
            for tier in TIERS:
                tier_data = model_df[model_df["tier"] == tier][metric].dropna().values
                if len(tier_data) > 0:
                    means.append(np.mean(tier_data))
                    sems.append(stats.sem(tier_data))
                else:
                    means.append(np.nan)
                    sems.append(0)

            ax.errorbar(
                tier_x, means, yerr=sems,
                marker='o', linewidth=2.5, markersize=10,
                label=model_name, color=colors.get(model_name, 'gray'),
                capsize=5, capthick=2
            )

        # Add Spearman correlation annotation
        if model_name in stat_results and metric in stat_results[model_name]:
            sp = stat_results[model_name][metric]["spearman"]
            if not np.isnan(sp["r"]):
                ax.text(0.5, 0.05, f"Spearman ρ = {sp['r']:.3f}\n(pooled models)",
                        transform=ax.transAxes, fontsize=9, ha='center',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel("Consistency Score", fontsize=10)
        ax.set_xticks(tier_x)
        ax.set_xticklabels([TIER_LABELS[t] for t in TIERS], fontsize=9)
        ax.set_ylim(0.3, 1.1)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3, label='Perfect consistency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "complexity_trend.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/complexity_trend.png")


def plot_answer_vs_reasoning_consistency(all_df):
    """Scatter plot: answer agreement vs reasoning chain similarity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Answer Agreement vs Reasoning Chain Similarity", fontsize=13, fontweight='bold')

    tier_colors = {'simple': '#4CAF50', 'moderate': '#FF9800', 'complex': '#F44336'}
    tier_markers = {'simple': 'o', 'moderate': 's', 'complex': '^'}

    for ax, model_name in zip(axes, MODELS):
        model_df = all_df[all_df["model"] == model_name].copy()
        if model_df.empty:
            ax.set_visible(False)
            continue

        for tier in TIERS:
            tier_data = model_df[model_df["tier"] == tier]
            if not tier_data.empty:
                ax.scatter(
                    tier_data["answer_agreement_rate"],
                    tier_data["reasoning_similarity"],
                    c=tier_colors[tier],
                    marker=tier_markers[tier],
                    label=f"{tier.capitalize()} (n={len(tier_data)})",
                    alpha=0.6, s=60, edgecolors='gray', linewidths=0.5
                )

        # Pearson correlation
        valid = model_df[["answer_agreement_rate", "reasoning_similarity"]].dropna()
        if len(valid) > 3:
            r, p = stats.pearsonr(valid["answer_agreement_rate"], valid["reasoning_similarity"])
            ax.text(0.05, 0.95, f"Pearson r = {r:.3f}, p = {p:.3f}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel("Answer Agreement Rate", fontsize=10)
        ax.set_ylabel("Reasoning Chain Similarity", fontsize=10)
        ax.set_title(f"{model_name}", fontsize=11, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "answer_vs_reasoning_consistency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/answer_vs_reasoning_consistency.png")


def plot_model_comparison(all_df):
    """Bar chart comparing haiku vs sonnet consistency across tiers."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_subset = ["answer_agreement_rate", "original_vs_paraphrase_agreement"]
    x_positions = np.arange(len(TIERS))
    width = 0.15
    colors_m = {'claude-haiku': '#2196F3', 'claude-sonnet': '#FF9800'}
    colors_metric = [0.7, 1.0]  # Alpha for each metric

    for m_idx, metric in enumerate(metrics_subset):
        for model_idx, model_name in enumerate(MODELS):
            model_df = all_df[all_df["model"] == model_name]
            means = []
            stds = []
            for tier in TIERS:
                vals = model_df[model_df["tier"] == tier][metric].dropna().values
                means.append(np.mean(vals) if len(vals) > 0 else 0)
                stds.append(np.std(vals) if len(vals) > 0 else 0)

            offset = (m_idx * len(MODELS) + model_idx) * width - (len(metrics_subset) * len(MODELS) - 1) * width / 2
            bars = ax.bar(
                x_positions + offset, means, width,
                yerr=stds,
                label=f"{model_name} ({metric.replace('_', ' ')})",
                color=colors_m[model_name],
                alpha=colors_metric[m_idx],
                error_kw=dict(capsize=3),
            )

    ax.set_xlabel("Complexity Tier", fontsize=11)
    ax.set_ylabel("Consistency Score", fontsize=11)
    ax.set_title("Model Comparison: Answer Consistency by Complexity Tier", fontsize=12, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 0))
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/model_comparison.png")


def main():
    all_dfs = []
    for model_name in MODELS:
        responses = load_responses(model_name)
        if not responses:
            continue
        print(f"\nAnalyzing {model_name}: {len(responses)} problems")
        df = analyze_model(responses, model_name)
        all_dfs.append(df)
        print(f"  Computed metrics for {len(df)} problems (with 3+ versions)")

    if not all_dfs:
        print("No response data found!")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal records: {len(all_df)}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    metrics = [
        "answer_agreement_rate", "pairwise_answer_agreement",
        "original_vs_paraphrase_agreement", "reasoning_similarity"
    ]

    summary_records = []
    for model_name in MODELS:
        for tier in TIERS:
            subset = all_df[(all_df["model"] == model_name) & (all_df["tier"] == tier)]
            if subset.empty:
                continue
            row = {"model": model_name, "tier": tier, "n": len(subset)}
            for m in metrics:
                vals = subset[m].dropna()
                if len(vals) > 0:
                    row[f"{m}_mean"] = np.mean(vals)
                    row[f"{m}_std"] = np.std(vals)
            summary_records.append(row)
            print(f"\n{model_name} | {tier} (n={len(subset)})")
            for m in metrics:
                vals = subset[m].dropna()
                if len(vals) > 0:
                    print(f"  {m}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(RESULTS_DIR / "summary_statistics.csv", index=False)

    # Statistical tests
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS")
    print("=" * 70)

    stat_results = {}
    for model_name in MODELS:
        model_df = all_df[all_df["model"] == model_name]
        if not model_df.empty:
            print(f"\n--- {model_name} ---")
            test_results = run_statistical_tests(model_df)
            stat_results[model_name] = test_results
            for metric, result in test_results.items():
                kw = result["kruskal_wallis"]
                sp = result["spearman"]
                print(f"\n{metric}:")
                print(f"  Kruskal-Wallis: H={kw['stat']:.4f}, p={kw['p_value']:.4f}")
                print(f"  Spearman ρ={sp['r']:.4f}, p={sp['p_value']:.4f}")
                print(f"  Means: {result['means_by_tier']}")
                print(f"  N per tier: {result['n_by_tier']}")

    # Save all data
    all_df.to_csv(RESULTS_DIR / "all_metrics.csv", index=False)
    with open(RESULTS_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_consistency_by_tier(all_df)
    plot_complexity_trend(all_df, stat_results)
    plot_answer_vs_reasoning_consistency(all_df)
    plot_model_comparison(all_df)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
