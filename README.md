# Qwen2.5-3B Coder: QLoRA Fine-Tuning & Local GGUF Edge Deployment

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Card-yellow)](https://huggingface.co)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Framework: Unsloth](https://img.shields.io/badge/Accelerated%20by-Unsloth%20Triton-purple)](https://github.com/unslothai/unsloth)
[![Inference: Ollama](https://img.shields.io/badge/Inference-Ollama%20%2F%20llama.cpp-black)](https://ollama.com)

An end-to-end pipeline fine-tuning `Qwen2.5-3B-Instruct` into an ultra-concise, non-thinking code generation assistant, compiled and quantized to 8-bit GGUF (`q8_0`) for zero-dependency local CLI inference via Ollama.

---

## 📌 Executive Summary

Standard base instruct models frequently produce conversational padding, scratchpad reasoning steps, and boilerplate sign-offs. This project implements parameter-efficient instruction alignment using **QLoRA** and custom Triton kernels, training across 250 steps to align model outputs to clean, non-thinking Python code.

The fine-tuned adapter is fused with the base model, converted into the modern **GGUF v3** binary format, and packaged into an Ollama runtime container for edge execution on laptop hardware.

---

## 🛠️ Architecture & Training Specifications

| Component | Specification |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-3B-Instruct` |
| **Quantization Precision** | 4-bit NormalFloat (NF4) with Double Quantization |
| **PEFT Configuration** | LoRA ($r=16, \alpha=32$, Dropout = 0) |
| **Target Modules** | All 7 linear projection layers (`q, k, v, o, gate, up, down`) |
| **Trainable Parameters** | **29.9M** out of 3.09B total (**0.96%** footprint) |
| **Context Length** | 1024 tokens |
| **Hardware** | Dual Tesla T4 GPUs (via Kaggle) |
| **Acceleration** | Fused Triton Cross-Entropy & Attention kernels (Unsloth) |
| **Effective Batch Size** | 8 ($2 \text{ per-device batch} \times 4 \text{ gradient accumulation}$) |
| **Optimizer & Schedule** | AdamW 8-bit, Cosine Decay, Warmup: 10 steps, Peak LR: $2\times 10^{-4}$ |
| **Training Steps** | 250 optimization steps |
| **Export Quantization** | GGUF 8-bit (`q8_0`, ~3.5 GB) |

---

## 📊 Dataset & Formatting

The model was aligned using a curated subset of **`Modotte/CodeX-7M-Non-Thinking`**:
* **Filtering:** Retained prompt-response pairs with clean code solutions while filtering out conversational preambles and thinking chains.
* **Schema Enforcement:** Standardized across Qwen's native **ChatML** token delimiters:

```text
