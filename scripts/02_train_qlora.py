"""Fine-tunes Qwen2.5-3B-Instruct using Unsloth QLoRA with Triton kernels."""

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MAX_SEQ_LENGTH = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")


def format_fn(examples):
  texts = [
      tokenizer.apply_chat_template(
          c, tokenize=False, add_generation_prompt=False
      )
      for c in examples["messages"]
  ]
  return {"text": texts}


raw_dataset = load_dataset(
    "json",
    data_files={"train": "data/train.jsonl", "eval": "data/eval.jsonl"},
)
train_dataset = raw_dataset["train"].map(format_fn, batched=True)

training_args = SFTConfig(
    output_dir="./models/checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    max_steps=250,
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    logging_steps=10,
    eval_strategy="no",
    save_strategy="no",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=training_args,
)

trainer.train()
model.save_pretrained("./models/adapter")
tokenizer.save_pretrained("./models/adapter")