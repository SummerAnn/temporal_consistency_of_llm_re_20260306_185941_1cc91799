# Datasets

This directory contains datasets for the research project on "Temporal Consistency of LLM Reasoning Under Paraphrased Prompts". Data files are NOT committed to git due to size. Follow the download instructions below.

## Overview

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | openai/gsm8k | 7473 train / 1319 test | Math word problems | datasets/gsm8k/ | Primary benchmark |
| MATH-500 | HuggingFaceH4/MATH-500 | 500 test | Competition math | datasets/math500/ | High-difficulty subset |
| GSM-Plus | qintongli/GSM-Plus | 10552 test / 2400 testmini | Adversarial math | datasets/gsm_plus/ | Paraphrase/perturbation benchmark |
| SVAMP | ChilleD/SVAMP | 700 train / 300 test | Math word problems | datasets/svamp/ | Simple variations |
| ARC-Challenge | allenai/ai2_arc | 1119 train / 1172 test | Science MCQ | datasets/arc/ | Multi-step reasoning |
| AQuA-RAT | aqua_rat | 97467 train / 254 test | Algebraic QA | datasets/aqua/ | With rationales |

## Recommended Primary Datasets

For studying temporal consistency under paraphrased prompts, we recommend:

1. **GSM8K** (primary): Well-studied, 7K+ math problems with solutions
2. **GSM-Plus** (perturbation analysis): Same problems with 8 types of variations including paraphrasing
3. **MATH-500** (complexity scaling): Competition-level problems for testing harder tasks
4. **SVAMP** (variation robustness): Designed to test sensitivity to simple linguistic changes

## Download Instructions

### GSM8K
```python
from datasets import load_dataset
dataset = load_dataset('openai/gsm8k', 'main')
dataset.save_to_disk('datasets/gsm8k')
```

### MATH-500
```python
from datasets import load_dataset
dataset = load_dataset('HuggingFaceH4/MATH-500')
dataset.save_to_disk('datasets/math500')
```

### GSM-Plus
```python
from datasets import load_dataset
dataset = load_dataset('qintongli/GSM-Plus')
dataset.save_to_disk('datasets/gsm_plus')
```

### SVAMP
```python
from datasets import load_dataset
dataset = load_dataset('ChilleD/SVAMP')
dataset.save_to_disk('datasets/svamp')
```

### ARC-Challenge
```python
from datasets import load_dataset
dataset = load_dataset('allenai/ai2_arc', 'ARC-Challenge')
dataset.save_to_disk('datasets/arc')
```

### AQuA-RAT
```python
from datasets import load_dataset
dataset = load_dataset('aqua_rat', 'raw')
dataset.save_to_disk('datasets/aqua')
```

## Loading Datasets

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk('datasets/gsm8k')
train = dataset['train']
test = dataset['test']
```

## GSM-Plus Perturbation Types

GSM-Plus is particularly valuable for this research as it provides 8 perturbation categories:
1. **Numerical Substitution** - Same problem structure, different numbers
2. **Digit Expansion** - Numbers expressed differently
3. **Integer-Decimal-Fraction Conversion** - Number format changes
4. **Adding Operation** - Extra arithmetic operation
5. **Reversing Operation** - Reversed operations
6. **Problem Understanding** - Text paraphrasing (most relevant to our hypothesis)
7. **Distractor Insertion** - Irrelevant sentence added
8. **Critical Thinking** - Incomplete/misleading problems

The "Problem Understanding" perturbation type is the most directly relevant to studying paraphrase consistency.
