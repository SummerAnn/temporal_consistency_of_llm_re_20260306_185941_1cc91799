"""
Query LLMs with all paraphrase versions of each problem.
Uses chain-of-thought prompting.
Records full reasoning chains and final answers.
Models tested: claude-haiku-4-5 and claude-sonnet-4-6
"""

import json
import os
import time
import re
from pathlib import Path
import anthropic

WORKSPACE = Path("/workspaces/temporal_consistency_of_llm_re_20260306_185941_1cc91799")
RESULTS_DIR = WORKSPACE / "results" / "data"

# Models to test
MODELS = [
    {"id": "claude-haiku-4-5-20251001", "name": "claude-haiku"},
    {"id": "claude-sonnet-4-6", "name": "claude-sonnet"},
]

COT_PROMPT_TEMPLATE = """Solve the following math problem step by step. Show all your work clearly.
After your reasoning, provide your final answer in a box like this: **Answer: [your answer]**

Problem:
{problem}

Solve step by step:"""


def call_claude_with_retry(client, model_id, prompt, max_retries=3, delay=5, temperature=0.0):
    """Call Claude API with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"    Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                print(f"    API error: {e}, retrying...")
                time.sleep(delay)
            else:
                raise


def extract_answer(response_text):
    """Extract the final answer from a CoT response."""
    # Try to find "Answer: [value]" pattern
    patterns = [
        r'\*\*Answer:\s*([^\*\n]+)\*\*',
        r'Answer:\s*([^\n]+)',
        r'\*\*Final Answer:\s*([^\*\n]+)\*\*',
        r'Final Answer:\s*([^\n]+)',
        r'\*\*\$([^$\*]+)\$\*\*',  # LaTeX in bold
        r'\\boxed\{([^}]+)\}',  # LaTeX boxed
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If no pattern matched, take the last non-empty line as a fallback
    lines = [l.strip() for l in response_text.split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return ""


def query_model_on_problem(client, model_id, paraphrase_entry):
    """Query a model on all versions of a problem and record responses."""
    results = []

    for version_idx, (version_text, version_label) in enumerate(
        zip(paraphrase_entry["versions"], paraphrase_entry["version_labels"])
    ):
        prompt = COT_PROMPT_TEMPLATE.format(problem=version_text)

        response = call_claude_with_retry(
            client, model_id, prompt, temperature=0.0
        )

        answer = extract_answer(response)

        results.append({
            "version_idx": version_idx,
            "version_label": version_label,
            "question": version_text,
            "full_response": response,
            "extracted_answer": answer,
        })

        time.sleep(0.3)  # Small delay between calls

    return results


def main():
    print("Loading paraphrases...")
    with open(RESULTS_DIR / "paraphrases.json") as f:
        paraphrases = json.load(f)

    print(f"Loaded {len(paraphrases)} problem sets")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    for model_info in MODELS:
        model_id = model_info["id"]
        model_name = model_info["name"]

        output_path = RESULTS_DIR / f"responses_{model_name}.json"

        # Load existing responses for resumability
        if output_path.exists():
            with open(output_path) as f:
                all_responses = json.load(f)
            done_ids = {r["problem_id"] for r in all_responses}
        else:
            all_responses = []
            done_ids = set()

        to_process = [p for p in paraphrases if p["problem_id"] not in done_ids]

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({model_id})")
        print(f"Remaining problems: {len(to_process)}/{len(paraphrases)}")
        print('='*60)

        if not to_process:
            print("All done for this model!")
            continue

        for i, para_entry in enumerate(to_process):
            print(f"\n[{i+1}/{len(to_process)}] Problem {para_entry['problem_id']} ({para_entry['tier']})")
            n_versions = len(para_entry["versions"])
            print(f"  Querying {n_versions} versions...")

            try:
                results = query_model_on_problem(client, model_id, para_entry)

                response_entry = {
                    "problem_id": para_entry["problem_id"],
                    "tier": para_entry["tier"],
                    "reference_answer": para_entry["reference_answer"],
                    "model_id": model_id,
                    "model_name": model_name,
                    "version_responses": results,
                }

                # Quick consistency check
                answers = [r["extracted_answer"] for r in results]
                print(f"  Answers: {answers}")

                all_responses.append(response_entry)

                # Save after each problem
                with open(output_path, "w") as f:
                    json.dump(all_responses, f, indent=2)

            except Exception as e:
                print(f"  ERROR: {e}")
                # Save what we have and continue
                with open(output_path, "w") as f:
                    json.dump(all_responses, f, indent=2)
                continue

        print(f"\nDone with {model_name}! Responses saved to {output_path}")


if __name__ == "__main__":
    main()
