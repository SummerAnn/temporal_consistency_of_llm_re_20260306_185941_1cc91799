# Cloned Repositories

5 repositories cloned for this research project.

## Repositories

### 1. PromptBench (Microsoft)
- **URL**: https://github.com/microsoft/promptbench
- **Location**: code/promptbench/
- **Purpose**: Unified LLM evaluation framework with adversarial prompt testing
- **Key files**:
  - `promptbench/` - main package
  - `examples/` - usage examples
  - `requirements.txt` - dependencies
- **Key features**:
  - Adversarial prompt attacks (character/word/sentence/semantic levels)
  - Dynamic evaluation (DyVal) to avoid data contamination
  - Prompt engineering methods (CoT, Emotion Prompt, Expert Prompting)
  - Supports many LLMs and datasets
- **Installation**: `pip install promptbench`
- **Use for our research**: Baseline adversarial evaluation; test model robustness to prompt variations

### 2. GSM-Plus
- **URL**: https://github.com/qtli/GSM-Plus
- **Location**: code/gsm_plus/
- **Purpose**: Adversarial math benchmark with 8 perturbation types for GSM8K
- **Key files**:
  - `scripts/` - evaluation scripts
  - `dataset/` - perturbation data
  - `results/` - results from prior experiments
- **Perturbation types**:
  - Numerical substitution, digit expansion, int-decimal-fraction conversion
  - Adding/reversing operations
  - Problem understanding (paraphrase) - most relevant
  - Distractor insertion, critical thinking
- **Use for our research**: Generate paraphrased versions of math problems; baseline for paraphrase robustness testing

### 3. ProSA (OpenCompass)
- **URL**: https://github.com/open-compass/ProSA
- **Location**: code/prosa/
- **Purpose**: Prompt sensitivity assessment framework (EMNLP 2024)
- **Key files**:
  - `prompts/` - prompt templates for sensitivity evaluation
  - `README.md` - detailed usage instructions
- **Key metric**: PromptSensiScore (PSS) - instance-level sensitivity measure
- **Use for our research**: Measure prompt sensitivity at instance level; compare our findings with PSS

### 4. GSM-IC (Google Research)
- **URL**: https://github.com/google-research-datasets/GSM-IC
- **Location**: code/gsm_ic/
- **Purpose**: GSM8K with irrelevant context (distractors) added
- **Key files**:
  - `GSM-IC_2step.json` - 2-step problems with irrelevant context
  - `GSM-IC_mstep.json` - multi-step problems with irrelevant context
- **Use for our research**: Test model robustness to irrelevant context (one type of prompt variation)

### 5. GSM8K-Consistency
- **URL**: https://github.com/SuperBruceJia/GSM8K-Consistency
- **Location**: code/gsm8k_consistency/
- **Purpose**: Benchmark database for analyzing consistency of arithmetic reasoning
- **Key files**:
  - `gsm8k_validation.jsonl` - validation data with consistency annotations
- **Use for our research**: Reference implementation for consistency measurement on GSM8K

## Usage Notes

### PromptBench Quick Start
```python
import promptbench as pb

# Load a model
model = pb.LLMModel(model='gpt-3.5-turbo', max_new_tokens=256, temperature=0)

# Load dataset
dataset = pb.DatasetLoader.load_dataset('gsm8k')

# Evaluate
for data in dataset:
    input_text = pb.InputProcess.basic_format(data['content'], None)
    output = model(input_text)
```

### GSM-Plus Loading
```python
from datasets import load_dataset
ds = load_dataset('qintongli/GSM-Plus')
# Filter for paraphrase perturbations
paraphrase_items = [x for x in ds['test'] if x['perturbation_type'] == 'problem_understanding']
```

### GSM-IC Loading
```python
import json
with open('code/gsm_ic/GSM-IC_mstep.json') as f:
    data = json.load(f)
```
