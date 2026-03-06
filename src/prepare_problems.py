"""
Prepare problem sets for the temporal consistency experiment.
Selects 30 problems per complexity tier (simple/moderate/complex).
- Simple: GSM-Plus "problem_understanding" pairs (grade school math)
- Moderate: MATH-500 problems at difficulty levels 1-2
- Complex: MATH-500 problems at difficulty levels 3-5
"""

import json
import random
from pathlib import Path
from datasets import load_from_disk

random.seed(42)

WORKSPACE = Path("/workspaces/temporal_consistency_of_llm_re_20260306_185941_1cc91799")
RESULTS_DIR = WORKSPACE / "results" / "data"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_PER_TIER = 30


def load_simple_problems():
    """Load GSM8K simple problems using GSM-Plus problem_understanding pairs."""
    gsm_plus = load_from_disk(str(WORKSPACE / "datasets" / "gsm_plus"))
    paraphrase_pairs = [
        x for x in gsm_plus["test"]
        if x["perturbation_type"] == "problem understanding"
    ]
    sampled = random.sample(paraphrase_pairs, N_PER_TIER)
    problems = []
    for i, p in enumerate(sampled):
        problems.append({
            "id": f"simple_{i:03d}",
            "tier": "simple",
            "original_question": p["seed_question"],
            "reference_answer": str(p["seed_answer"]),
            "known_paraphrase": p["question"],  # Pre-made paraphrase from GSM-Plus
            "perturbation_type": "problem_understanding",
            "source": "gsm_plus",
        })
    return problems


def load_moderate_problems():
    """Load MATH-500 problems at difficulty levels 1-2 (moderate complexity)."""
    math500 = load_from_disk(str(WORKSPACE / "datasets" / "math500"))
    # Filter for level 1-2 problems (numeric levels)
    moderate = [
        x for x in math500["test"]
        if x.get("level") in [1, 2]
    ]
    print(f"  Found {len(moderate)} moderate MATH-500 problems (Level 1-2)")
    sampled = random.sample(moderate, min(N_PER_TIER, len(moderate)))
    problems = []
    for i, p in enumerate(sampled):
        problems.append({
            "id": f"moderate_{i:03d}",
            "tier": "moderate",
            "original_question": p["problem"],
            "reference_answer": p["answer"],
            "known_paraphrase": None,
            "perturbation_type": None,
            "source": "math500",
            "level": p.get("level"),
            "subject": p.get("subject", "unknown"),
        })
    return problems


def load_complex_problems():
    """Load MATH-500 problems at difficulty levels 3-5 (complex)."""
    math500 = load_from_disk(str(WORKSPACE / "datasets" / "math500"))
    complex_probs = [
        x for x in math500["test"]
        if x.get("level") in [3, 4, 5]
    ]
    print(f"  Found {len(complex_probs)} complex MATH-500 problems (Level 3-5)")
    sampled = random.sample(complex_probs, min(N_PER_TIER, len(complex_probs)))
    problems = []
    for i, p in enumerate(sampled):
        problems.append({
            "id": f"complex_{i:03d}",
            "tier": "complex",
            "original_question": p["problem"],
            "reference_answer": p["answer"],
            "known_paraphrase": None,
            "perturbation_type": None,
            "source": "math500",
            "level": p.get("level"),
            "subject": p.get("subject", "unknown"),
        })
    return problems


def main():
    print("Loading and preparing problem sets...")

    print("Loading simple problems (GSM8K / GSM-Plus)...")
    simple = load_simple_problems()
    print(f"  Loaded {len(simple)} simple problems")

    print("Loading moderate problems (MATH-500 Level 1-2)...")
    moderate = load_moderate_problems()
    print(f"  Loaded {len(moderate)} moderate problems")

    print("Loading complex problems (MATH-500 Level 3-5)...")
    complex_probs = load_complex_problems()
    print(f"  Loaded {len(complex_probs)} complex problems")

    all_problems = simple + moderate + complex_probs
    print(f"\nTotal problems: {len(all_problems)}")

    # Save problems
    output_path = RESULTS_DIR / "problems.json"
    with open(output_path, "w") as f:
        json.dump(all_problems, f, indent=2)
    print(f"\nSaved {len(all_problems)} problems to {output_path}")

    # Print summary
    from collections import Counter
    tier_counts = Counter(p["tier"] for p in all_problems)
    print(f"Tier distribution: {dict(tier_counts)}")

    # Show sample
    print("\nSample simple problem:")
    print(f"  Q: {simple[0]['original_question'][:100]}...")
    print(f"  A: {simple[0]['reference_answer']}")
    print("\nSample moderate problem:")
    print(f"  Q: {moderate[0]['original_question'][:100]}...")
    print(f"  Level: {moderate[0].get('level')}, Subject: {moderate[0].get('subject')}")
    print("\nSample complex problem:")
    print(f"  Q: {complex_probs[0]['original_question'][:100]}...")
    print(f"  Level: {complex_probs[0].get('level')}, Subject: {complex_probs[0].get('subject')}")


if __name__ == "__main__":
    main()
