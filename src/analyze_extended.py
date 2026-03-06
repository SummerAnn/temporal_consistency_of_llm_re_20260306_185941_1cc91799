"""
Extended analysis with improved answer normalization and error categorization.
Two normalization approaches:
1. Strict: full string normalization (may penalize formatting differences)
2. Numeric-focused: extract first number/mathematical expression (for simple problems)
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from scipy import stats

WORKSPACE = Path("/workspaces/temporal_consistency_of_llm_re_20260306_185941_1cc91799")
RESULTS_DIR = WORKSPACE / "results" / "data"
PLOTS_DIR = WORKSPACE / "results" / "plots"
TIERS = ["simple", "moderate", "complex"]
MODELS = ["claude-haiku", "claude-sonnet"]


def normalize_strict(answer):
    """Full string normalization."""
    if not answer:
        return ""
    answer = re.sub(r'\\[a-zA-Z]+\{([^}]+)\}', r'\1', answer)
    answer = answer.replace('$', '').replace('\\', '').replace(',', '')
    answer = re.sub(r'^[\[\(](.+)[\]\)]$', r'\1', answer.strip())
    answer = re.sub(r'\s+', '', answer.lower()).rstrip('.')
    for prefix in ['answer:', 'finalanswer:', 'theanswer']:
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
    return answer.strip()


def extract_numeric_or_math(answer):
    """
    Extract numeric/mathematical core from answer.
    For word problems: extracts the first number.
    For math problems: extracts mathematical expressions.
    This handles formatting variations like '45 fairies' vs '45'.
    """
    if not answer:
        return ""

    # Clean basic formatting
    text = answer.replace('$', '').replace('\\', '').replace(',', '').strip()

    # Try to extract LaTeX boxed answer
    boxed = re.search(r'boxed\{([^}]+)\}', text)
    if boxed:
        return re.sub(r'\s+', '', boxed.group(1).lower())

    # Try to extract fraction
    frac = re.search(r'frac\{([^}]+)\}\{([^}]+)\}', text)
    if frac:
        return f"{frac.group(1)}/{frac.group(2)}"

    # Extract leading number (handles "45 fairies" -> "45", "21 years old" -> "21")
    # But for math, don't just grab first number
    first_num = re.match(r'^[-−]?[\d\.]+', text.strip())
    if first_num:
        return first_num.group(0)

    # Extract any number pattern (including decimals, negative)
    nums = re.findall(r'[-−]?[\d,]+\.?\d*', text)
    if nums:
        return nums[0].replace(',', '')

    # Fall back to cleaned text
    return re.sub(r'\s+', '', text.lower()).rstrip('.')[:50]


def compute_paa_both(answers):
    """Compute pairwise agreement with both normalization methods."""
    if len(answers) < 2:
        return 1.0, 1.0

    strict = [normalize_strict(a) for a in answers]
    numeric = [extract_numeric_or_math(a) for a in answers]

    n = len(answers)
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    paa_strict = sum(1 for i, j in pairs if strict[i] == strict[j]) / len(pairs)
    paa_numeric = sum(1 for i, j in pairs if numeric[i] == numeric[j]) / len(pairs)

    return float(paa_strict), float(paa_numeric)


def classify_inconsistency_type(answers):
    """
    Classify what type of inconsistency exists.
    Returns: 'consistent', 'formatting_only', 'factual_inconsistency'
    """
    strict = [normalize_strict(a) for a in answers if a]
    numeric = [extract_numeric_or_math(a) for a in answers if a]

    if not strict or not numeric:
        return 'unknown'

    # Check if all strictly equal
    unique_strict = set(strict)
    unique_numeric = set(numeric)

    if len(unique_strict) == 1:
        return 'consistent'
    elif len(unique_numeric) == 1:
        return 'formatting_only'  # Only formatting differs, not the number
    else:
        return 'factual_inconsistency'  # Actual answer disagrees


def run_analysis_with_both_norms():
    """Run full analysis with both normalization methods."""

    all_records = []

    for model_name in MODELS:
        path = RESULTS_DIR / f"responses_{model_name}.json"
        if not path.exists():
            continue
        with open(path) as f:
            responses = json.load(f)

        for entry in responses:
            if len(entry.get("version_responses", [])) < 3:
                continue

            answers = [v["extracted_answer"] for v in entry["version_responses"]]
            tier = entry["tier"]

            paa_strict, paa_numeric = compute_paa_both(answers)
            inc_type = classify_inconsistency_type(answers)

            all_records.append({
                "problem_id": entry["problem_id"],
                "tier": tier,
                "model": model_name,
                "paa_strict": paa_strict,
                "paa_numeric": paa_numeric,
                "inconsistency_type": inc_type,
                "n_versions": len(answers),
            })

    df = pd.DataFrame(all_records)
    return df


def plot_normalization_comparison(df):
    """Show how strict vs numeric normalization affects results."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Impact of Normalization Method on Answer Consistency\n"
        "(Strict vs Numeric - the difference reveals formatting-only variation)",
        fontsize=12, fontweight='bold'
    )

    tier_x = [0, 1, 2]
    tier_labels = {
        "simple": "Simple\n(GSM8K)",
        "moderate": "Moderate\n(MATH L1-2)",
        "complex": "Complex\n(MATH L3-5)"
    }
    colors = {'claude-haiku': '#2196F3', 'claude-sonnet': '#FF9800'}
    ls_map = {'paa_strict': '--', 'paa_numeric': '-'}
    marker_map = {'paa_strict': 's', 'paa_numeric': 'o'}

    for ax, model_name in zip(axes, MODELS):
        model_df = df[df["model"] == model_name]
        if model_df.empty:
            continue

        for norm_col, label_suffix in [("paa_strict", "Strict"), ("paa_numeric", "Numeric (core answer)")]:
            means = []
            sems = []
            for tier in TIERS:
                vals = model_df[model_df["tier"] == tier][norm_col].dropna().values
                means.append(np.mean(vals) if len(vals) > 0 else np.nan)
                sems.append(stats.sem(vals) if len(vals) > 1 else 0)

            ax.errorbar(
                tier_x, means, yerr=sems,
                marker=marker_map[norm_col],
                linestyle=ls_map[norm_col],
                linewidth=2, markersize=9,
                label=f"{label_suffix} normalization",
                color='#333333' if norm_col == 'paa_strict' else '#e74c3c',
                capsize=4
            )

        ax.set_title(f"{model_name}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Pairwise Answer Agreement", fontsize=10)
        ax.set_xticks(tier_x)
        ax.set_xticklabels([tier_labels[t] for t in TIERS], fontsize=9)
        ax.set_ylim(0.3, 1.1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "normalization_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/normalization_comparison.png")


def plot_inconsistency_types(df):
    """Stacked bar chart of inconsistency types by tier and model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Types of Inconsistency by Complexity Tier", fontsize=13, fontweight='bold')

    type_colors = {
        'consistent': '#4CAF50',
        'formatting_only': '#FF9800',
        'factual_inconsistency': '#F44336',
        'unknown': '#9E9E9E',
    }
    type_order = ['consistent', 'formatting_only', 'factual_inconsistency']

    for ax, model_name in zip(axes, MODELS):
        model_df = df[df["model"] == model_name]
        if model_df.empty:
            continue

        x = np.arange(len(TIERS))
        bottom = np.zeros(len(TIERS))

        type_counts = {}
        for inc_type in type_order:
            counts = []
            for tier in TIERS:
                tier_df = model_df[model_df["tier"] == tier]
                n = len(tier_df)
                if n > 0:
                    count = (tier_df["inconsistency_type"] == inc_type).sum()
                    counts.append(count / n)
                else:
                    counts.append(0)
            type_counts[inc_type] = counts

            bars = ax.bar(
                x, counts, bottom=bottom,
                color=type_colors[inc_type],
                label=inc_type.replace('_', ' ').capitalize(),
                alpha=0.85, width=0.5
            )
            bottom += np.array(counts)

        ax.set_title(f"{model_name}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Fraction of Problems", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(["Simple\n(GSM8K)", "Moderate\n(MATH L1-2)", "Complex\n(MATH L3-5)"], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "inconsistency_types.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/plots/inconsistency_types.png")


def main():
    print("Running extended analysis with dual normalization...")
    df = run_analysis_with_both_norms()
    print(f"Total records: {len(df)}")

    print("\n" + "=" * 70)
    print("SUMMARY: Strict vs Numeric Normalization")
    print("=" * 70)

    records = []
    for model_name in MODELS:
        for tier in TIERS:
            subset = df[(df["model"] == model_name) & (df["tier"] == tier)]
            if subset.empty:
                continue

            strict_mean = subset["paa_strict"].mean()
            numeric_mean = subset["paa_numeric"].mean()
            formatting_gap = numeric_mean - strict_mean

            inc_counts = subset["inconsistency_type"].value_counts()
            pct_consistent = inc_counts.get("consistent", 0) / len(subset)
            pct_formatting = inc_counts.get("formatting_only", 0) / len(subset)
            pct_factual = inc_counts.get("factual_inconsistency", 0) / len(subset)

            print(f"\n{model_name} | {tier} (n={len(subset)})")
            print(f"  PAA (strict):  {strict_mean:.4f}")
            print(f"  PAA (numeric): {numeric_mean:.4f}")
            print(f"  Formatting gap: {formatting_gap:.4f}")
            print(f"  Consistent: {pct_consistent:.1%}")
            print(f"  Formatting-only inconsistency: {pct_formatting:.1%}")
            print(f"  Factual inconsistency: {pct_factual:.1%}")

            records.append({
                "model": model_name, "tier": tier, "n": len(subset),
                "paa_strict": strict_mean, "paa_numeric": numeric_mean,
                "formatting_gap": formatting_gap,
                "pct_consistent": pct_consistent,
                "pct_formatting_only": pct_formatting,
                "pct_factual": pct_factual,
            })

    result_df = pd.DataFrame(records)
    result_df.to_csv(RESULTS_DIR / "extended_analysis.csv", index=False)

    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (Numeric Normalization)")
    print("=" * 70)
    for model_name in MODELS:
        model_df = df[df["model"] == model_name]
        groups = {t: model_df[model_df["tier"] == t]["paa_numeric"].dropna().values for t in TIERS}
        valid = [g for g in groups.values() if len(g) > 0]
        if len(valid) >= 2:
            try:
                stat, p = stats.kruskal(*valid)
                print(f"\n{model_name} - Kruskal-Wallis (numeric): H={stat:.4f}, p={p:.4f}")
            except Exception:
                pass
            # Spearman
            tier_order = {"simple": 0, "moderate": 1, "complex": 2}
            tier_idx = model_df["tier"].map(tier_order)
            mask = model_df["paa_numeric"].notna()
            if mask.sum() > 2:
                r, p = stats.spearmanr(tier_idx[mask], model_df["paa_numeric"][mask])
                print(f"  Spearman ρ={r:.4f}, p={p:.4f}")

    print("\nGenerating plots...")
    plot_normalization_comparison(df)
    plot_inconsistency_types(df)

    print("\nExtended analysis complete!")


if __name__ == "__main__":
    main()
