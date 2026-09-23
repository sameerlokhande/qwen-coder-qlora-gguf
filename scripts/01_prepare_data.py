"""Streams, cleans, and structures instruction dataset into Qwen ChatML schema."""

import os
from datasets import Dataset, load_dataset

DATASET_NAME = "Modotte/CodeX-7M-Non-Thinking"
SAMPLE_SIZE = 5000
SYSTEM_PROMPT = (
    "You are Qwen, an expert programming assistant. "
    "Provide clear, bug-free, and well-structured code solutions."
)

os.makedirs("data", exist_ok=True)
streamed_dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

records = []
for row in streamed_dataset:
  prompt = row.get("input")
  code = row.get("output")
  if not prompt or not code:
    continue
  if len(prompt.strip()) < 15 or len(code.strip()) < 15:
    continue

  records.append({
      "messages": [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": prompt.strip()},
          {"role": "assistant", "content": code.strip()},
      ]
  })
  if len(records) >= SAMPLE_SIZE:
    break

splits = Dataset.from_list(records).train_test_split(test_size=0.05, seed=42)
splits["train"].to_json("data/train.jsonl", orient="records", lines=True)
splits["test"].to_json("data/eval.jsonl", orient="records", lines=True)
print(f"Data saved: {len(splits['train'])} train, {len(splits['test'])} eval.")