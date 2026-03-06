# Literature Review: Temporal Consistency of LLM Reasoning Under Paraphrased Prompts

**Research Hypothesis**: LLMs produce inconsistent reasoning chains when given semantically equivalent but syntactically different prompts, and this inconsistency increases with task complexity.

---

## Research Area Overview

The problem of LLM reasoning consistency under paraphrased inputs is a well-established and actively growing research area. The central empirical finding across the literature is unambiguous: LLMs are sensitive to surface-form variations in prompts, even when the semantic content is identical. This sensitivity manifests in two ways relevant to our hypothesis: (1) **output accuracy changes** across paraphrased versions of the same task, and (2) **reasoning chain structure changes** even when final answers are the same. The literature further shows that this inconsistency is not fully explained by stochastic sampling, is more pronounced in complex tasks, and is not simply resolved by scaling model size.

---

## Key Papers

### 1. Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification (PPCV)
- **Authors**: Weili Shi, Dongliang Guo, Lehan Yang, Tianlong Wang, Hanzhang Yuan, Sheng Li (UVA)
- **Year**: 2026 (preprint Feb 2026)
- **Source**: arXiv:2602.11361
- **Key Contribution**: PPCV framework using paraphrastic consistency as a signal for identifying "critical tokens" in reasoning chains
- **Methodology**:
  - Stage 1: Generate paraphrased forms using APE-optimized prompts; identify positions where token predictions diverge between original and paraphrased inputs (these are "critical tokens")
  - Stage 2: Generate alternative tokens at critical positions; select final answer by consistency across paraphrased variants
  - Key insight: >90% of correct rollouts achieve consistency score ≥1 across paraphrases; only ~30% of incorrect rollouts do
- **Datasets**: GSM8K, GSM-Hard, Math500, SVAMP, ARC, AIME2024, AIME2025, BRUMO2025, HMMT2025
- **Models**: Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, Qwen3-32B
- **Results**: PPCV substantially outperforms CoT, Self-Consistency, Tree-of-Thought, Guided Decoding, Predictive Decoding, and Phi-Decoding. On Llama-3.1: GSM8K 77.40→88.24%, Math500 31.00→50.00%, SVAMP 83.00→89.60%
- **Relevance**: Directly demonstrates that paraphrase consistency distinguishes correct from incorrect reasoning; provides methodology and benchmarks

### 2. Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing (PRIN)
- **Authors**: Jihyun Janice Ahn, Wenpeng Yin (Penn State)
- **Year**: 2025 (COLM 2025)
- **Source**: arXiv:2504.01282
- **Key Contribution**: Discovers PRIN - a new inconsistency type: LLMs give contradictory answers to "which are correct?" vs "which are incorrect?" versions of the same judgment task
- **Methodology**: Direct vs Reverse prompting; tested on 6 LLMs across 3 tasks at varying difficulty levels
- **Datasets**: MATH, MathQA, EquationInference
- **Models**: GPT-4, Llama-3-8B, Llama-3.3-70B, Falcon-40B, Qwen2.5-72B, Mixtral-8x22B-MoE
- **Results**: PRIN does not correlate with Randomness Inconsistency or Paraphrase Inconsistency; some low-paraphrase-sensitivity models still show high PRIN; explicit CoT reduces PRIN
- **Relevance**: Shows LLM inconsistency is multi-dimensional; provides taxonomy of inconsistency types relevant to our study

### 3. Semantic Consistency for Assuring Reliability of Large Language Models
- **Authors**: Multiple
- **Year**: 2023
- **Source**: arXiv:2308.09138
- **Key Contribution**: Framework for measuring semantic consistency using paraphrase-generated answers; A2C prompting strategy
- **Methodology**: AuxLLM generates paraphrases; main LLM answers each; pairwise semantic equivalence computed; consistency metric aggregated
- **Datasets**: Various QA benchmarks
- **Results**: Consistent answers across paraphrases are more likely to be correct; A2C strategy improves over majority voting
- **Relevance**: Establishes semantic consistency as a reliability signal; foundational methodology

### 4. Exploring LLM Reasoning Through Controlled Prompt Variations
- **Authors**: Multiple
- **Year**: 2025
- **Source**: arXiv:2504.02111
- **Key Contribution**: Systematic study of 4 perturbation categories on 13 LLMs using GSM8K
- **Methodology**: Four perturbation types: irrelevant context, pathological instructions, relevant non-essential context, combinations
- **Datasets**: GSM8K
- **Models**: 13 open-source and closed-source LLMs
- **Results**: Irrelevant context significantly degrades performance; degradation NOT correlated with task complexity (number of reasoning steps); degradation NOT correlated with model size
- **Relevance**: Direct test of our hypothesis that complexity increases inconsistency - interestingly finds NO correlation with reasoning step count, but complexity measured differently than we might

### 5. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs
- **Authors**: Apple Research
- **Year**: 2024 (ICLR 2025)
- **Source**: arXiv:2410.05229
- **Key Contribution**: Symbolic templates allow controllable generation of math variants; reveals LLM performance is surprisingly fragile
- **Methodology**: Generate multiple problem instances from symbolic templates by varying names, numbers, and structure; test performance across variants
- **Datasets**: GSM-Symbolic (based on GSM8K)
- **Results**: All models decline when numbers change; performance drops up to 65% when a single irrelevant clause is added; more clauses = worse performance; suggests pattern matching not genuine reasoning
- **Relevance**: Shows inconsistency increases with structural complexity (clause count); partially supports our hypothesis

### 6. GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs
- **Authors**: Qintong Li, Leyang Cui, Xueliang Zhao, Lingpeng Kong, Wei Bi
- **Year**: 2024
- **Source**: arXiv:2402.19255
- **Key Contribution**: 8-perturbation-type extension of GSM8K; most comprehensive math robustness benchmark
- **Methodology**: For each GSM8K test problem, create 8 variations; test 25 LLMs with 4 prompting techniques
- **Datasets**: GSM-Plus (10552 test / 2400 testmini)
- **Results**: LLMs solve original but fail on variations; none of studied prompting techniques achieve consistent performance; distractor insertion most damaging; paraphrase (problem understanding) also significant
- **Relevance**: Primary benchmark for paraphrase robustness in math reasoning; 25 LLMs provide broad comparison

### 7. POSIX: A Prompt Sensitivity Index For Large Language Models
- **Authors**: Multiple
- **Year**: 2024
- **Source**: arXiv:2410.02185
- **Key Contribution**: POSIX metric based on relative change in log-likelihood of response when prompt is replaced with intent-preserving variant
- **Methodology**: Capture relative log-likelihood change; not solely based on performance metrics
- **Results**: Scaling model size does NOT reduce sensitivity; single few-shot example almost always reduces sensitivity; paraphrasing causes highest sensitivity in open-ended tasks; template changes cause highest sensitivity in MCQ tasks
- **Relevance**: Provides a metric for measuring our research's key variable (prompt sensitivity/inconsistency)

### 8. ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs (EMNLP 2024)
- **Authors**: Jingming Zhuo et al. (OpenCompass/Shanghai AI Lab)
- **Year**: 2024 (EMNLP 2024 Findings)
- **Source**: arXiv:2410.12405
- **Key Contribution**: PromptSensiScore (PSS) - instance-level sensitivity metric using decoding confidence
- **Methodology**: Generate 3 prompt versions per instance; measure sensitivity using PSS; correlate with model confidence
- **Datasets**: AlpacaEval, Arena Hard Auto, various task benchmarks
- **Results**: Sensitivity varies across datasets and models; larger models more robust; few-shot reduces sensitivity; complex reasoning tasks most sensitive; higher model confidence → more robust
- **Relevance**: Instance-level sensitivity measure aligns with our goal of studying per-problem consistency

### 9. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts
- **Authors**: Kaijie Zhu et al. (Microsoft)
- **Year**: 2023
- **Source**: arXiv:2306.04528
- **Key Contribution**: Comprehensive adversarial prompt benchmark across 4 attack levels and 8 tasks
- **Methodology**: 4788 adversarial prompts; character/word/sentence/semantic attacks mimicking plausible user errors
- **Datasets**: 13 datasets covering NLI, QA, translation, math
- **Results**: All LLMs show significant sensitivity to adversarial prompts across all attack levels; semantic attacks most effective
- **Relevance**: Establishes baseline adversarial robustness measurements; methodology for multi-level perturbation analysis

### 10. Self-Consistency Improves Chain of Thought Reasoning in Language Models
- **Authors**: Xuezhi Wang, Jason Wei, Dale Schuurmans, et al.
- **Year**: 2023 (ICLR)
- **Source**: arXiv:2203.11171
- **Key Contribution**: Self-consistency decoding: sample diverse CoT paths, select most consistent answer by majority vote
- **Methodology**: Sample N diverse reasoning paths at temperature > 0; marginalize over paths via majority voting
- **Datasets**: GSM8K, SVAMP, AQuA, StrategyQA, ARC-challenge
- **Results**: +17.9% on GSM8K, +11.0% on SVAMP, +12.2% on AQuA, +6.4% on StrategyQA, +3.9% on ARC
- **Code Available**: Yes (via promptbench and many reimplementations)
- **Relevance**: Primary baseline method for comparison; establishes that diverse CoT paths can be aggregated for better answers

### 11. Can Language Models Perform Robust Reasoning in CoT with Noisy Rationales?
- **Authors**: Zhanke Zhou et al.
- **Year**: 2024
- **Source**: arXiv:2410.23856
- **Key Contribution**: NoRa dataset for evaluating CoT robustness to noise; CD-CoT mitigation method
- **Methodology**: Add irrelevant or inaccurate thoughts to CoT examples; measure performance drop
- **Datasets**: NoRa (based on multiple benchmarks)
- **Results**: LLMs drop 1.4-19.8% with irrelevant thoughts, 2.2-40.4% with inaccurate thoughts; self-consistency shows limited efficacy; CD-CoT improves by 17.8% on average
- **Relevance**: Shows chain robustness depends on noise type; inaccurate thoughts worse than irrelevant ones

### 12. Consistency in Language Models: Current Landscape, Challenges, and Future Directions
- **Authors**: Multiple
- **Year**: 2025
- **Source**: arXiv:2505.00268
- **Key Contribution**: Survey of LLM consistency research; taxonomy and gaps
- **Key findings**:
  - Semantic consistency testing commonly uses automatic paraphrasing methods
  - Transitive/symmetric consistency and non-English languages are underrepresented
  - Semantic consistency of English text-to-text models is relatively well-studied
- **Relevance**: Survey provides framework for classifying our research; identifies gaps we can fill

---

## Common Methodologies

### Paraphrase Generation
- **APE (Automatic Prompt Engineering)**: Used in PPCV to generate paraphrases that maximize solve rate diversity (Zhou et al., 2022)
- **LLM-generated paraphrases**: Ask model to rephrase while preserving mathematical relationships and numerical values
- **Template-based variation**: GSM-Symbolic, GSM-Plus generate controlled variations with known perturbation types
- **Human annotations**: Some datasets use human-written paraphrases for higher quality

### Consistency Measurement
- **Majority voting**: Self-Consistency (Wang et al., 2023) - simple and effective baseline
- **Semantic equivalence**: Check if answers are semantically equivalent, not just string-equal
- **Consistency score**: PPCV counts how many paraphrases agree with original; >90% correct answers score ≥1
- **POSIX**: Log-likelihood relative change for intent-preserving prompt replacements
- **PromptSensiScore (PSS)**: Instance-level sensitivity using decoding confidence

### Evaluation Frameworks
- **PromptBench**: Unified framework for adversarial prompt evaluation
- **ProSA**: Prompt sensitivity assessment with PSS metric
- **LM Evaluation Harness**: Standard LLM evaluation framework

---

## Standard Baselines

- **Chain-of-Thought (CoT)**: Wei et al. (2022) - generate intermediate reasoning steps; standard baseline
- **Self-Consistency**: Wang et al. (2023) - sample multiple CoT paths + majority vote; strong baseline (+17.9% on GSM8K)
- **Tree-of-Thought**: Yao et al. (2023) - tree search over reasoning trajectories
- **Guided Decoding**: Constrained generation approaches
- **Predictive Decoding**: Probing future trajectories to guide current step
- **Phi-Decoding**: Adaptive foresight sampling (Xu et al., 2025)
- **Zero-shot CoT**: "Let's think step by step" (Kojima et al., 2022)

---

## Evaluation Metrics

| Metric | When to Use | Notes |
|--------|-------------|-------|
| Accuracy | Primary measure | Simple correct/incorrect on final answer |
| Pass@k | Sampling-based | Probability of correct in k samples |
| Consistency Rate | Cross-paraphrase agreement | % of paraphrases with matching answers |
| PromptSensiScore (PSS) | Instance-level sensitivity | Uses decoding confidence |
| POSIX | Prompt sensitivity index | Log-likelihood ratio based |
| Performance Spread | Robustness measure | Max - min accuracy across prompt variations |

---

## Datasets in the Literature

| Dataset | Task | # Examples | Used in | Notes |
|---------|------|-----------|---------|-------|
| GSM8K | Grade school math | 7473 train / 1319 test | Most papers | Primary benchmark |
| MATH/Math500 | Competition math | 12500 total / 500 test subset | PPCV, PRIN | High difficulty |
| SVAMP | Math word problems | 1000 | PPCV, Self-Consistency | Tests robustness to linguistic variations |
| ARC-Challenge | Science MCQ | 2590 | PPCV, PromptRobust | Multi-step reasoning |
| AQuA-RAT | Algebraic QA | 97467 train / 254 test | Self-Consistency | Multiple choice with rationales |
| GSM-Plus | Adversarial GSM8K | 10552 test | GSM-Plus paper | 8 perturbation types |
| GSM-Symbolic | Symbolic GSM8K | Variable | GSM-Symbolic paper | Controllable generation |
| GSM-IC | GSM8K + irrelevant context | Variable | Distractor studies | 2-step and multi-step variants |
| MATH (Hendrycks) | Competition math | 12500 | Multiple | Various subjects |
| MathQA | Algebraic reasoning | ~37K | PRIN | Multiple choice |
| EquationInference | Equation reasoning | Variable | PRIN | PhD-level difficulty |
| AIME 2024/2025 | Competition math | 30 per year | PPCV | Extreme difficulty |

---

## Gaps and Opportunities

1. **Reasoning chain analysis beyond answer accuracy**: Most work measures answer consistency, but not how the reasoning chains themselves differ. Our research directly addresses this.

2. **Cross-task complexity analysis**: GSM-Symbolic finds performance drops with clause count, but this is different from task type complexity. Need systematic study across task types.

3. **Consistency as a function of paraphrase type**: Most work treats paraphrasing as binary; few papers systematically vary the degree of syntactic vs semantic change.

4. **Longitudinal/temporal consistency**: No work specifically studies whether consistency changes over time (e.g., across model versions or training stages) - this is our novel angle.

5. **Intermediate step consistency**: PPCV identifies critical tokens but doesn't fully measure how much reasoning chains differ structurally. A systematic metric for reasoning chain similarity would be novel.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **GSM8K** (primary): Standard, well-understood, 1319 test examples
2. **GSM-Plus** (paraphrase study): Built-in "problem_understanding" perturbation type isolates paraphrase effect; 8 types allow comparison
3. **MATH-500** (complexity study): Test consistency on harder problems to validate complexity hypothesis
4. **SVAMP** (linguistic variation baseline): Designed to test sensitivity to simple linguistic changes

### Recommended Baselines
1. **Standard CoT** (zero-shot and few-shot): Baseline reasoning
2. **Self-Consistency** (Wang et al., 2023): Strong sampling-based baseline
3. **Temperature-0 greedy decoding**: Deterministic baseline to isolate non-stochastic inconsistency

### Recommended Metrics
1. **Answer consistency rate** across N paraphrases: Primary metric (% agreement)
2. **Reasoning chain similarity**: Novel metric measuring structural similarity of CoT chains (e.g., BERTScore, edit distance on steps)
3. **Paraphrase-accuracy correlation**: Does accuracy on paraphrased version predict accuracy on original?
4. **POSIX or PSS**: If white-box access to models available

### Experimental Design Recommendations
- Generate multiple paraphrase levels (syntactic-only, mixed, semantic-only) to tease apart sources of inconsistency
- Use GSM-Plus "problem_understanding" perturbation type as a controlled paraphrase set
- Test across at least 3 task complexity levels (GSM8K = easy, MATH-500 = hard, AIME = extreme)
- Compare both closed-source (API) and open-source models to identify patterns
- Analyze not just whether answers differ, but how reasoning chains differ at the step level

### Methodological Considerations
- Use APE to optimize paraphrasing prompts (as done in PPCV) rather than naive paraphrasing
- Control for numerical values when generating paraphrases to isolate linguistic vs numerical sensitivity
- Consider that "correct reasoning paths are more robust to paraphrastic perturbations" (PPCV finding) can be used as both a measurement framework and a correctness signal
- The PPCV finding that paraphrase consistency distinguishes correct from incorrect answers suggests consistency measurement can serve as a quality proxy
