# Research Planning: Temporal Consistency of LLM Reasoning Under Paraphrased Prompts

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly deployed in scientific, medical, and high-stakes domains where reasoning reliability is critical. If semantically identical questions yield different reasoning chains and/or answers, the reliability of LLM-based scientific tools is fundamentally compromised. Understanding when and why inconsistency occurs enables better deployment practices and identifies which task types are most vulnerable.

### Gap in Existing Work
Based on literature_review.md:
- Most work measures *answer* consistency, not *reasoning chain structural similarity*
- Studies on complexity vs consistency have conflicting results (GSM-Symbolic shows clause count matters; arXiv:2504.02111 finds NO correlation with reasoning step count - different measures of complexity)
- No systematic study generating multiple paraphrase types for same problems across a difficulty spectrum and comparing reasoning chain similarity (not just final answers)
- Most prior work is on math; applying to multiple task domains at once is underexplored

### Our Novel Contribution
We provide a systematic empirical study that: (1) measures both answer consistency AND reasoning chain semantic similarity across paraphrase variants; (2) tests across three clearly defined complexity tiers (simple/moderate/complex tasks); (3) uses multiple Claude LLM variants; (4) analyzes the correlation between task complexity and consistency degradation with statistical rigor.

### Experiment Justification
- **Experiment 1 (Paraphrase generation)**: Needed to create controlled variants; using Claude to generate 4 paraphrases per problem with constrained prompts ensures quality
- **Experiment 2 (Multi-model querying)**: Test 2 Claude models (haiku vs sonnet) across all paraphrases to see if model scale affects consistency
- **Experiment 3 (Complexity scaling analysis)**: Compare consistency across simple/moderate/complex tiers to test our core hypothesis
- **Experiment 4 (Reasoning chain similarity)**: Novel analysis of semantic similarity of CoT chains using sentence embeddings, beyond just answer agreement

---

## Research Question
Do LLMs produce inconsistent reasoning chains when given semantically equivalent but syntactically different prompts, and does this inconsistency increase with task complexity?

## Background and Motivation
LLM reasoning sensitivity to prompt phrasing has been documented (GSM-Symbolic, GSM-Plus, PromptRobust), but the relationship between paraphrasing and *reasoning chain consistency* is underexplored. Prior work focuses on answer accuracy. Understanding reasoning-level inconsistency matters for scientific reliability.

## Hypothesis Decomposition

1. **H1 (Answer Inconsistency)**: LLMs produce different final answers for semantically equivalent paraphrases of the same task, at a rate higher than random baseline
2. **H2 (Reasoning Chain Inconsistency)**: Even when final answers agree, reasoning chains differ structurally (measured by semantic similarity)
3. **H3 (Complexity Scaling)**: Both answer inconsistency and reasoning chain inconsistency increase with task complexity (simple < moderate < complex)
4. **H4 (Model Size Effect)**: Larger models (claude-sonnet) show lower inconsistency than smaller models (claude-haiku)

## Proposed Methodology

### Approach
Use pre-gathered datasets (GSM8K/GSM-Plus for simple, MATH-500 for complex) combined with LLM-generated paraphrases to create controlled paraphrase sets. Query two Claude models with all paraphrase variants and measure both answer consistency and reasoning chain semantic similarity.

### Experimental Steps

1. **Problem selection (30 per tier)**:
   - Tier 1 (Simple): 30 GSM8K problems from GSM-Plus "problem_understanding" pairs
   - Tier 2 (Moderate): 30 MATH-500 problems at levels 1-2
   - Tier 3 (Complex): 30 MATH-500 problems at levels 3-5

2. **Paraphrase generation**: For each of the 90 problems, generate 4 additional paraphrases using Claude claude-sonnet-4-6, plus the original = 5 versions per problem

3. **LLM querying**: Query claude-haiku-4-5-20251001 and claude-sonnet-4-6 on all 5 versions of each problem with chain-of-thought prompting (450 calls per model, 900 total for the test set)

4. **Consistency measurement**:
   - Answer agreement rate: % of 5 responses with same answer
   - Reasoning chain semantic similarity: cosine similarity of sentence embeddings for reasoning chains

5. **Statistical analysis**: Kruskal-Wallis test across tiers; Spearman correlation between complexity proxy and consistency; effect sizes

### Baselines
- **Random baseline**: Expected agreement by chance for each answer type
- **Same-prompt resampling**: Run same prompt 5 times (temperature=0.7) to measure stochastic inconsistency baseline

### Evaluation Metrics
- **Answer Agreement Rate (AAR)**: Fraction of paraphrases where the extracted answer matches the mode (majority)
- **Pairwise Answer Agreement (PAA)**: Pairwise comparison - what % of pairs agree
- **Reasoning Semantic Similarity (RSS)**: Mean pairwise cosine similarity of sentence embeddings of reasoning chains
- **Consistency Degradation Slope**: Linear regression slope of consistency vs complexity tier

### Statistical Analysis Plan
- Kruskal-Wallis H-test across 3 complexity tiers (non-parametric, doesn't assume normality)
- Post-hoc Mann-Whitney U tests with Bonferroni correction for pairwise tier comparisons
- Spearman correlation between complexity level and consistency metrics
- Significance threshold: α = 0.05

## Expected Outcomes
- H1: Yes, LLMs show non-trivial answer inconsistency (based on GSM-Plus literature showing ~10-30% drop)
- H2: Yes, reasoning chains differ even when answers agree (novel finding)
- H3: Inconsistency increases with complexity (supported by GSM-Symbolic findings, but contested by arXiv:2504.02111)
- H4: Larger models are more consistent (supported by ProSA findings)

## Timeline and Milestones
1. Environment setup + data preparation: 30 min
2. Paraphrase generation for 90 problems: 20 min
3. LLM querying (900 API calls): 40 min
4. Analysis and visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- API rate limits: Use retries with exponential backoff
- Answer extraction: Use regex + LLM extraction for complex math answers
- Missing API keys for OpenAI: Use only Anthropic models (haiku + sonnet)

## Success Criteria
- Statistically significant difference in consistency across complexity tiers (p < 0.05)
- Clear visualization of complexity-consistency relationship
- Novel finding about reasoning chain similarity vs answer similarity
