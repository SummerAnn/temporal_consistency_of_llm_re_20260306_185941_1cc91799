"""
Generate paraphrases for each problem using Claude claude-haiku-4-5.
For each of the 90 problems, generates 4 paraphrases.
The original question + 4 paraphrases = 5 versions per problem.
Paraphrases preserve mathematical content and numerical values.
"""

import json
import os
import time
import random
from pathlib import Path
import anthropic

random.seed(42)

WORKSPACE = Path("/workspaces/temporal_consistency_of_llm_re_20260306_185941_1cc91799")
RESULTS_DIR = WORKSPACE / "results" / "data"

# Use Claude Haiku for paraphrase generation (fast and cheap)
PARAPHRASE_MODEL = "claude-haiku-4-5-20251001"
N_PARAPHRASES = 4  # We'll have 1 original + 4 generated = 5 total

PARAPHRASE_PROMPT = """You are a helpful assistant that paraphrases math problems.
Your task is to rewrite the following math problem in {n} different ways.

CRITICAL RULES:
1. Preserve ALL numerical values exactly (no changes to numbers)
2. Preserve ALL mathematical relationships and operations
3. Change sentence structure, word choice, and phrasing significantly
4. Each paraphrase must be semantically identical but syntactically different
5. Each paraphrase should read naturally
6. Do not add or remove any information

Original problem:
{problem}

Generate exactly {n} paraphrased versions. Format your response as a JSON array:
[
  "paraphrase 1 here",
  "paraphrase 2 here",
  ...
]

Only output the JSON array, nothing else."""


def call_claude_with_retry(client, model, messages, max_retries=3, delay=5):
    """Call Claude API with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=messages,
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


def generate_paraphrases_for_problem(client, problem, n=N_PARAPHRASES):
    """Generate n paraphrases for a given problem."""
    prompt = PARAPHRASE_PROMPT.format(
        n=n,
        problem=problem["original_question"]
    )

    response_text = call_claude_with_retry(
        client,
        PARAPHRASE_MODEL,
        [{"role": "user", "content": prompt}]
    )

    # Parse JSON response
    try:
        # Try to extract JSON array from response
        text = response_text.strip()
        if text.startswith("["):
            paraphrases = json.loads(text)
        else:
            # Find JSON array in response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                paraphrases = json.loads(text[start:end])
            else:
                print(f"    Warning: Could not parse JSON from response")
                paraphrases = []
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        paraphrases = []

    # Ensure we have the right number
    if len(paraphrases) < n:
        print(f"    Warning: Got {len(paraphrases)}/{n} paraphrases")

    return paraphrases[:n]


def main():
    print("Loading problems...")
    with open(RESULTS_DIR / "problems.json") as f:
        problems = json.load(f)

    print(f"Loaded {len(problems)} problems")

    # Check if we can load existing paraphrases (for resumability)
    paraphrases_path = RESULTS_DIR / "paraphrases.json"
    if paraphrases_path.exists():
        with open(paraphrases_path) as f:
            all_paraphrases = json.load(f)
        print(f"Loaded {len(all_paraphrases)} existing paraphrase sets")
        # Find which problems still need paraphrases
        done_ids = {p["problem_id"] for p in all_paraphrases}
        problems_to_process = [p for p in problems if p["id"] not in done_ids]
        print(f"Still need to generate paraphrases for {len(problems_to_process)} problems")
    else:
        all_paraphrases = []
        problems_to_process = problems

    if not problems_to_process:
        print("All paraphrases already generated!")
        return

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    print(f"\nGenerating {N_PARAPHRASES} paraphrases for each of {len(problems_to_process)} problems...")
    print(f"Model: {PARAPHRASE_MODEL}")

    for i, problem in enumerate(problems_to_process):
        print(f"\n[{i+1}/{len(problems_to_process)}] Problem {problem['id']} ({problem['tier']})")
        print(f"  Q: {problem['original_question'][:80]}...")

        # Use known_paraphrase from GSM-Plus if available
        known_paraphrases = []
        if problem.get("known_paraphrase"):
            known_paraphrases = [problem["known_paraphrase"]]
            n_to_generate = N_PARAPHRASES - 1  # Already have 1
        else:
            n_to_generate = N_PARAPHRASES

        # Generate additional paraphrases
        generated = generate_paraphrases_for_problem(client, problem, n=n_to_generate)
        print(f"  Generated {len(generated)} paraphrases")

        paraphrases = known_paraphrases + generated

        # Create all 5 versions: original + paraphrases
        all_versions = [problem["original_question"]] + paraphrases[:N_PARAPHRASES]

        result = {
            "problem_id": problem["id"],
            "tier": problem["tier"],
            "reference_answer": problem["reference_answer"],
            "versions": all_versions,  # 5 versions: [original, p1, p2, p3, p4]
            "version_labels": ["original"] + [f"paraphrase_{j+1}" for j in range(len(paraphrases[:N_PARAPHRASES]))],
        }
        all_paraphrases.append(result)

        # Save after each problem for resumability
        with open(paraphrases_path, "w") as f:
            json.dump(all_paraphrases, f, indent=2)

        # Small delay to avoid rate limits
        time.sleep(0.5)

    print(f"\n\nDone! Generated paraphrases for {len(all_paraphrases)} problems")
    print(f"Saved to {paraphrases_path}")

    # Print summary stats
    total_versions = sum(len(p["versions"]) for p in all_paraphrases)
    print(f"Total prompt versions created: {total_versions}")


if __name__ == "__main__":
    main()
