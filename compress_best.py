import json
from llmlingua import PromptCompressor

path = "/root/autodl-tmp/open-unlearning/saves/unlearn/SAMPLE_UNLEARN/temperature=0.1"
results = []

llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
)
print("Loading llm_lingua model...")
with open(path + "/generations_n30.json", 'r') as f:
    for line in f:
        if line.strip():
            item = json.loads(line)
            ground_truth = item.get('ground_truth', '')
            if not ground_truth:
                continue
            compressed_ground_truth = llm_lingua.compress_prompt(
                ground_truth, rate=0.75
            )
            print(f"Compressed ground_truth: {compressed_ground_truth}")
            results.append({
                'ground_truth': ground_truth,
                'compressed_ground_truth': compressed_ground_truth
            })

with open(path + "/compressed_ground_truth.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)