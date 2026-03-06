# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project on "Temporal Consistency of LLM Reasoning Under Paraphrased Prompts."

- **Papers downloaded**: 18
- **Datasets downloaded**: 6
- **Repositories cloned**: 5

---

## Papers

Total papers downloaded: 18

| Title | Year | arXiv ID | File | Key Info |
|-------|------|----------|------|----------|
| Finding the Cracks: PPCV | 2026 | 2602.11361 | papers/2602.11361_paraphrastic_probing_consistency.pdf | Most directly relevant; paraphrase consistency framework |
| Prompt-Reverse Inconsistency (PRIN) | 2025 | 2504.01282 | papers/2504.01282_prompt_reverse_inconsistency.pdf | New inconsistency type; COLM 2025 |
| Semantic Consistency for LLM Reliability | 2023 | 2308.09138 | papers/2308.09138_semantic_consistency_reliability.pdf | A2C prompting; consistency metric |
| Exploring LLM Reasoning Through Controlled Prompt Variations | 2025 | 2504.02111 | papers/2504.02111_controlled_prompt_variations.pdf | 4 perturbation types; 13 LLMs |
| GSM-Symbolic | 2024 | 2410.05229 | papers/2410.05229_gsm_symbolic.pdf | ICLR 2025; all models decline with variations |
| GSM-Plus | 2024 | 2402.19255 | papers/2402.19255_gsm_plus.pdf | 8 perturbation types; 25 LLMs |
| POSIX: Prompt Sensitivity Index | 2024 | 2410.02185 | papers/2410.02185_posix_prompt_sensitivity_index.pdf | Log-likelihood based sensitivity metric |
| ProSA: Prompt Sensitivity Assessment | 2024 | 2410.12405 | papers/2410.12405_prosa_prompt_sensitivity.pdf | EMNLP 2024; instance-level PSS metric |
| PromptRobust | 2023 | 2306.04528 | papers/2306.04528_promptrobust.pdf | 4788 adversarial prompts; Microsoft |
| Benchmarking Prompt Sensitivity | 2025 | 2502.06065 | papers/2502.06065_benchmarking_prompt_sensitivity.pdf | PromptSET dataset |
| Chain-of-Thought Prompting | 2022 | 2201.11903 | papers/2201.11903_chain_of_thought_prompting.pdf | Foundational CoT paper |
| Self-Consistency Improves CoT | 2023 | 2203.11171 | papers/2203.11171_self_consistency_cot.pdf | Primary baseline; ICLR; +17.9% GSM8K |
| Semantic Self-Consistency | 2024 | 2410.07839 | papers/2410.07839_semantic_self_consistency.pdf | Semantic-weighted majority voting |
| Robust CoT with Noisy Rationales | 2024 | 2410.23856 | papers/2410.23856_noisy_rationales_cot.pdf | NoRa dataset; CD-CoT method |
| Consistency in LLMs Survey | 2025 | 2505.00268 | papers/2505.00268_consistency_llm_survey.pdf | Survey; taxonomy; gaps |
| LLM Robustness to Prompt Format Styles | 2025 | 2504.06969 | papers/2504.06969_llm_robustness_prompt_format.pdf | Performance spread metric |
| Same Question, Different Words (LAP) | 2025 | 2503.01345 | papers/2503.01345_latent_adversarial_paraphrasing.pdf | 2x reward variance on semantically equivalent prompts |
| Prompt Repetition Improves Non-Reasoning LLMs | 2025 | 2512.14982 | papers/2512.14982_prompt_repetition.pdf | Prompt redundancy helps non-reasoning models |

See papers/README.md for detailed descriptions.

---

## Datasets

Total datasets downloaded: 6

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | openai/gsm8k (HF) | 7473 train / 1319 test | Grade school math | datasets/gsm8k/ | Primary benchmark; gold standard |
| MATH-500 | HuggingFaceH4/MATH-500 (HF) | 500 test | Competition math (high/college) | datasets/math500/ | High difficulty for complexity scaling study |
| GSM-Plus | qintongli/GSM-Plus (HF) | 10552 test / 2400 testmini | Adversarial math (8 perturbation types) | datasets/gsm_plus/ | Best paraphrase benchmark |
| SVAMP | ChilleD/SVAMP (HF) | 700 train / 300 test | Math word problems | datasets/svamp/ | Designed for linguistic variation testing |
| ARC-Challenge | allenai/ai2_arc (HF) | 1119 train / 1172 test | Science multiple-choice | datasets/arc/ | Multi-step commonsense reasoning |
| AQuA-RAT | aqua_rat (HF) | 97467 train / 254 test | Algebraic QA with rationales | datasets/aqua/ | MCQ with reasoning chains |

See datasets/README.md for detailed descriptions and download instructions.

### Key Dataset Feature: GSM-Plus Perturbation Types

GSM-Plus is the most valuable dataset for this research as it provides systematic perturbations:
- `numerical_substitution`: Changes numbers only
- `digit_expansion`: Different number format
- `integer_decimal_fraction`: Number type changes
- `adding_operation`: Extra operation
- `reversing_operation`: Reversed operations
- `problem_understanding`: **Text paraphrasing** (most relevant to our hypothesis)
- `distractor_insertion`: Irrelevant sentences
- `critical_thinking`: Incomplete/misleading information

---

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| PromptBench | github.com/microsoft/promptbench | Unified LLM adversarial evaluation | code/promptbench/ | pip install promptbench |
| GSM-Plus | github.com/qtli/GSM-Plus | Adversarial math benchmark + evaluation | code/gsm_plus/ | Evaluation scripts included |
| ProSA | github.com/open-compass/ProSA | Prompt sensitivity assessment (EMNLP 2024) | code/prosa/ | PSS metric; prompt templates |
| GSM-IC | github.com/google-research-datasets/GSM-IC | GSM8K with irrelevant context | code/gsm_ic/ | JSON data files ready to use |
| GSM8K-Consistency | github.com/SuperBruceJia/GSM8K-Consistency | Arithmetic reasoning consistency benchmark | code/gsm8k_consistency/ | Validation data included |

See code/README.md for detailed descriptions and usage instructions.

---

## Resource Gathering Notes

### Search Strategy

Manual search was conducted using web search (arXiv.org, Semantic Scholar) after paper-finder service was unavailable. Search queries included:
- "LLM reasoning consistency paraphrased prompts"
- "prompt sensitivity robustness LLM chain-of-thought"
- "LLM self-consistency semantic equivalence benchmark"
- "LLM reasoning task complexity prompt variation inconsistency"

### Selection Criteria

Papers were selected based on:
1. Direct relevance to paraphrase consistency or prompt sensitivity in LLM reasoning
2. Recent publication (2023-2026) for state-of-the-art coverage
3. High citation potential or published at major venues (ICLR, EMNLP, COLM, NeurIPS)
4. Papers providing benchmarks, datasets, or evaluation frameworks

Datasets were selected based on:
1. Standard reasoning benchmarks used across multiple papers
2. Availability of perturbation/paraphrase variants
3. Range of difficulty levels for testing complexity hypothesis

### Challenges Encountered

- Paper-finder service was not running; manual web search conducted
- wget not available in environment; Python requests used for downloads
- Some HuggingFace dataset names changed (MATH dataset requires different path)
- A few older datasets (StrategyQA, MathQA) use loading scripts no longer supported by HF

### Gaps and Workarounds

- **GSM-Hard not on HuggingFace**: Referenced in PPCV paper; downloadable from the GSM8K-Hard repository if needed. Alternative: use MATH-500 for hard problems.
- **AIME 2024/2025**: Small competition datasets (~30 problems each); available through various collections or manually assembled
- **GSM-Symbolic**: Generated on-demand from templates; Apple Research dataset available at apple/GSM-Symbolic on HF (not downloaded due to large size)

---

## Recommendations for Experiment Design

### 1. Primary Dataset: GSM-Plus with "problem_understanding" perturbation
This provides a ready-made controlled comparison: same problem, paraphrased text. Allows direct measurement of reasoning chain consistency.

```python
from datasets import load_from_disk
gsm_plus = load_from_disk('datasets/gsm_plus')
paraphrase_pairs = [(x, x['seed_question']) for x in gsm_plus['test']
                    if x['perturbation_type'] == 'problem_understanding']
```

### 2. Complexity Study: Compare GSM8K vs MATH-500
- GSM8K: 1-5 reasoning steps, grade school level → baseline difficulty
- MATH-500: Competition problems, multi-step → high difficulty
- This tests the hypothesis that inconsistency increases with task complexity

### 3. Paraphrase Generation: Use APE-optimized prompt (PPCV methodology)
- PPCV paper provides APE methodology for high-quality paraphrase generation
- Key constraint: preserve all numerical values and mathematical relationships
- Generate 5 paraphrases per problem (as done in PPCV)

### 4. Evaluation: Two-level consistency measurement
- **Answer consistency**: Jaccard similarity of final answers across paraphrases
- **Reasoning chain consistency**: Structural similarity of CoT steps (step count, operation types, intermediate values)

### 5. Models to Test
- **Closed-source**: GPT-4o, Claude 3.5 (via API for black-box testing)
- **Open-source**: Llama-3.1-8B, Llama-3.1-70B, Mistral-7B (for white-box analysis with token probabilities)
- Comparison across model families tests generalizability

### 6. Key Metrics
- **Consistency Rate (CR)**: Fraction of paraphrases matching original answer
- **Chain Edit Distance (CED)**: Minimum edits between reasoning chain structures
- **Complexity-Inconsistency Correlation**: Spearman correlation between problem difficulty and CR
