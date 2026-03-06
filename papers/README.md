# Downloaded Papers

18 papers downloaded covering LLM reasoning consistency, paraphrase robustness, and related benchmarks.

## Core Papers (Most Directly Relevant)

1. **[Finding the Cracks: Improving LLMs Reasoning with Paraphrastic Probing and Consistency Verification](2602.11361_paraphrastic_probing_consistency.pdf)**
   - Authors: Weili Shi, Dongliang Guo, Lehan Yang, et al.
   - Year: 2026
   - arXiv: 2602.11361
   - Why relevant: Directly proposes PPCV framework using paraphrastic consistency for LLM reasoning. Key empirical finding: correct reasoning paths are more robust to paraphrastic perturbations (>90% of correct rollouts have consistency score ≥1 vs ~30% for incorrect).

2. **[Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing](2504.01282_prompt_reverse_inconsistency.pdf)**
   - Authors: Jihyun Janice Ahn, Wenpeng Yin
   - Year: 2025 (COLM 2025)
   - arXiv: 2504.01282
   - Why relevant: Discovers PRIN - a new form of LLM inconsistency beyond paraphrase inconsistency. Tests on MATH, MathQA, EquationInference. Shows up to 10% variation even under deterministic settings.

3. **[Semantic Consistency for Assuring Reliability of Large Language Models](2308.09138_semantic_consistency_reliability.pdf)**
   - Authors: Multiple
   - Year: 2023
   - arXiv: 2308.09138
   - Why relevant: Framework for measuring/improving semantic consistency using paraphrased prompts. Introduces A2C prompting strategy and semantic consistency metric.

4. **[Exploring LLM Reasoning Through Controlled Prompt Variations](2504.02111_controlled_prompt_variations.pdf)**
   - Authors: Multiple
   - Year: 2025
   - arXiv: 2504.02111
   - Why relevant: Systematic study of reasoning robustness under 4 categories of perturbations on GSM8K. Tests 13 LLMs. Key finding: performance degradation not correlated with task complexity or model size.

## Benchmarking Papers

5. **[GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](2410.05229_gsm_symbolic.pdf)**
   - Authors: Apple Research
   - Year: 2024 (ICLR 2025)
   - arXiv: 2410.05229
   - Why relevant: Shows all LLMs decline when numerical values change; performance drops up to 65% with added clauses. Suggests pattern matching over genuine reasoning.

6. **[GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers](2402.19255_gsm_plus.pdf)**
   - Authors: Qintong Li et al.
   - Year: 2024
   - arXiv: 2402.19255
   - Why relevant: 8 perturbation types for GSM8K. Tests 25 LLMs. None of studied prompting techniques are sufficiently robust across all variations.

7. **[POSIX: A Prompt Sensitivity Index For Large Language Models](2410.02185_posix_prompt_sensitivity_index.pdf)**
   - Authors: Multiple
   - Year: 2024
   - arXiv: 2410.02185
   - Why relevant: Proposes POSIX metric for measuring prompt sensitivity. Paraphrasing causes highest sensitivity in open-ended tasks.

8. **[ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs](2410.12405_prosa_prompt_sensitivity.pdf)**
   - Authors: Jingming Zhuo et al. (OpenCompass)
   - Year: 2024 (EMNLP 2024)
   - arXiv: 2410.12405
   - Why relevant: PromptSensiScore (PSS) metric. Larger models more robust; few-shot reduces sensitivity; complex reasoning tasks most sensitive.

9. **[PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](2306.04528_promptrobust.pdf)**
   - Authors: Kaijie Zhu et al. (Microsoft)
   - Year: 2023
   - arXiv: 2306.04528
   - Why relevant: 4788 adversarial prompts across 8 tasks and 13 datasets; character/word/sentence/semantic levels.

10. **[Benchmarking Prompt Sensitivity in Large Language Models](2502.06065_benchmarking_prompt_sensitivity.pdf)**
    - Authors: Multiple
    - Year: 2025
    - arXiv: 2502.06065
    - Why relevant: PromptSET dataset for prompt sensitivity prediction.

## Foundational Methods

11. **[Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](2201.11903_chain_of_thought_prompting.pdf)**
    - Authors: Jason Wei et al. (Google)
    - Year: 2022
    - arXiv: 2201.11903
    - Why relevant: Foundational CoT paper; shows exemplar order affects accuracy (54% to 93% on SST-2).

12. **[Self-Consistency Improves Chain of Thought Reasoning in Language Models](2203.11171_self_consistency_cot.pdf)**
    - Authors: Xuezhi Wang et al.
    - Year: 2023 (ICLR)
    - arXiv: 2203.11171
    - Why relevant: Primary baseline; sample diverse CoT paths + majority vote. +17.9% on GSM8K.

13. **[Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting](2410.07839_semantic_self_consistency.pdf)**
    - Authors: Multiple
    - Year: 2024
    - arXiv: 2410.07839
    - Why relevant: Extends self-consistency by incorporating semantic information from reasoning paths.

## Robustness and Consistency Studies

14. **[Can Language Models Perform Robust Reasoning in Chain-of-Thought Prompting with Noisy Rationales?](2410.23856_noisy_rationales_cot.pdf)**
    - Authors: Zhanke Zhou et al.
    - Year: 2024
    - arXiv: 2410.23856
    - Why relevant: NoRa dataset; LLMs drop 1.4-40.4% accuracy with noisy rationales. Proposes CD-CoT.

15. **[Consistency in Language Models: Current Landscape, Challenges, and Future Directions](2505.00268_consistency_llm_survey.pdf)**
    - Authors: Multiple
    - Year: 2025
    - arXiv: 2505.00268
    - Why relevant: Survey of consistency research; taxonomy and gaps identified.

16. **[Towards LLMs Robustness to Changes in Prompt Format Styles](2504.06969_llm_robustness_prompt_format.pdf)**
    - Authors: Multiple
    - Year: 2025
    - arXiv: 2504.06969
    - Why relevant: Measures robustness via performance spread across format variations.

17. **[Same Question, Different Words: Latent Adversarial Framework for Prompt Robustness](2503.01345_latent_adversarial_paraphrasing.pdf)**
    - Authors: Multiple
    - Year: 2025
    - arXiv: 2503.01345
    - Why relevant: Best-case reward can be 2x worst-case for semantically equivalent prompts (Llama-2-13b on RobustAlpaca).

18. **[Prompt Repetition Improves Non-Reasoning LLMs](2512.14982_prompt_repetition.pdf)**
    - Authors: Multiple
    - Year: 2025
    - arXiv: 2512.14982
    - Why relevant: Related finding that prompt redundancy (repetition) helps non-reasoning models.
