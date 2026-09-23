"""Merges LoRA adapter into base weights and compiles to quantized GGUF."""

import glob
from huggingface_hub import HfApi, login
from unsloth import FastLanguageModel

EXPORT_DIR = "./models/gguf_export"
MAX_SEQ_LENGTH = 1024

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./models/adapter",
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model.save_pretrained_gguf(EXPORT_DIR, tokenizer, quantization_method="q8_0")

# Optional upload to Hugging Face Hub
api = HfApi()
repo_id = f"{api.whoami()['name']}/qwen2.5-3b-coder-q8_0"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

gguf_path = glob.glob(f"{EXPORT_DIR}/**/*.gguf", recursive=True)[0]
api.upload_file(
    path_or_fileobj=gguf_path,
    path_in_repo="qwen2.5-3b-instruct.Q8_0.gguf",
    repo_id=repo_id,
    repo_type="model",
)