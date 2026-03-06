# Temporal Consistency of LLM Reasoning Under Paraphrased Prompts

Research project investigating whether LLMs produce inconsistent reasoning chains when given semantically equivalent but syntactically different prompts, and whether this inconsistency scales with task complexity.

## Key Findings

- **LLMs are factually consistent for simple tasks, inconsistent for complex ones**: Simple GSM8K problems show near-zero factual inconsistency (0-6.7%), while MATH-500 Level 3-5 problems show 22-26% factual inconsistency
- **Complexity drives inconsistency**: Spearman ρ = -0.368, p = 0.001 (Haiku); the effect is statistically significant
- **Formatting vs. factual inconsistency**: Simple word problems create large formatting variation (e.g., "45 fairies" vs "45") that must be separated from true factual inconsistency — a critical methodological insight
- **Reasoning chains always vary**: TF-IDF similarity of reasoning chains is ~0.47-0.54 regardless of tier, showing that LLMs take different reasoning paths for paraphrased prompts even when final answers match
- **Larger model (Sonnet) is more consistent**: Claude Sonnet shows ~8-50% lower factual inconsistency than Haiku on moderate/complex tasks

## How to Reproduce

### Setup
```bash
# Create environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install anthropic scikit-learn scipy numpy pandas matplotlib tqdm datasets

# Set API key
export ANTHROPIC_API_KEY=your_key_here
```

### Run Experiments
```bash
# Phase 1: Prepare problem sets
python src/prepare_problems.py

# Phase 2: Generate paraphrases
python src/generate_paraphrases.py

# Phase 3: Query LLMs with all versions
python src/query_llms.py

# Phase 4: Analyze consistency
python src/analyze_consistency.py
python src/analyze_extended.py
```

### Expected Runtime
- Paraphrase generation: ~15 min (90 problems × 4 paraphrases via API)
- LLM querying: ~30-45 min per model (90 problems × 5 versions)
- Analysis: ~2 min

## File Structure

```
├── src/
│   ├── prepare_problems.py      # Problem selection from datasets
│   ├── generate_paraphrases.py  # LLM-based paraphrase generation
│   ├── query_llms.py            # Query models on all paraphrase versions
│   ├── analyze_consistency.py   # Main consistency analysis
│   └── analyze_extended.py      # Normalization comparison & type classification
├── results/
│   ├── data/
│   │   ├── problems.json             # 90 selected problems
│   │   ├── paraphrases.json          # 5 versions per problem
│   │   ├── responses_claude-haiku.json   # Haiku model responses
│   │   ├── responses_claude-sonnet.json  # Sonnet model responses
│   │   ├── all_metrics.csv           # Per-problem consistency metrics
│   │   ├── extended_analysis.csv     # Dual-normalization results
│   │   └── statistical_tests.json    # Statistical test outputs
│   └── plots/
│       ├── consistency_by_tier.png       # Box plots by tier/model
│       ├── complexity_trend.png          # Trend across complexity
│       ├── normalization_comparison.png  # Strict vs numeric normalization
│       ├── inconsistency_types.png       # Stacked type classification
│       └── answer_vs_reasoning_consistency.png
├── datasets/           # Pre-downloaded datasets (local only)
├── papers/             # Downloaded reference papers
├── planning.md         # Detailed research plan
├── REPORT.md           # Full research report with all findings
└── literature_review.md  # Pre-gathered literature review
```

## Full Details

See [REPORT.md](REPORT.md) for complete methodology, results tables, statistical tests, and discussion.
