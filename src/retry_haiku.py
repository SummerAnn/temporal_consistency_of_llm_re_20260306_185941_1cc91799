"""Retry missing haiku responses."""
import json, os, time, re
import anthropic

RESULTS_DIR = "results/data"
MODEL_ID = "claude-haiku-4-5-20251001"
MODEL_NAME = "claude-haiku"

COT_PROMPT_TEMPLATE = """Solve the following math problem step by step. Show all your work clearly.
After your reasoning, provide your final answer in a box like this: **Answer: [your answer]**

Problem:
{problem}

Solve step by step:"""


def extract_answer(response_text):
    patterns = [r'\*\*Answer:\s*([^\*\n]+)\*\*', r'Answer:\s*([^\n]+)', r'\\boxed\{([^}]+)\}']
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    lines = [l.strip() for l in response_text.split('\n') if l.strip()]
    return lines[-1] if lines else ""


def main():
    with open(f"{RESULTS_DIR}/paraphrases.json") as f:
        paraphrases = json.load(f)
    with open(f"{RESULTS_DIR}/responses_{MODEL_NAME}.json") as f:
        existing = json.load(f)
    done_ids = {r["problem_id"] for r in existing}
    missing = [p for p in paraphrases if p["problem_id"] not in done_ids]
    print(f"Missing: {len(missing)} problems")
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    for i, para_entry in enumerate(missing):
        print(f"[{i+1}/{len(missing)}] {para_entry['problem_id']}")
        results = []
        for v_idx, (v_text, v_label) in enumerate(zip(para_entry["versions"], para_entry["version_labels"])):
            try:
                resp = client.messages.create(model=MODEL_ID, max_tokens=1024, temperature=0.0,
                    messages=[{"role": "user", "content": COT_PROMPT_TEMPLATE.format(problem=v_text)}])
                response_text = resp.content[0].text
                answer = extract_answer(response_text)
            except Exception as e:
                print(f"  Error: {e}")
                response_text = ""; answer = ""
            results.append({"version_idx": v_idx, "version_label": v_label, "question": v_text,
                "full_response": response_text, "extracted_answer": answer})
            time.sleep(0.3)
        print(f"  Answers: {[r['extracted_answer'] for r in results]}")
        existing.append({"problem_id": para_entry["problem_id"], "tier": para_entry["tier"],
            "reference_answer": para_entry["reference_answer"], "model_id": MODEL_ID,
            "model_name": MODEL_NAME, "version_responses": results})
        with open(f"{RESULTS_DIR}/responses_{MODEL_NAME}.json", "w") as f:
            json.dump(existing, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
