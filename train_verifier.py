#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA训练数值校验器（RQ2）。
    python train_verifier.py data/verifier/train.jsonl data/verifier/val.jsonl \
        -o models/verifier_lora
基座：Qwen2.5-7B-Instruct（4bit nf4量化 + LoRA r16），4060 Ti 16GB可跑。
训练目标：对话式 prompt -> 单标签补全（"支持"/"不支持"），仅对标签token计loss。
类别均衡：训练集"支持"样本上采样3倍（原始正负比约1:4）。
"""
import argparse
import json
import random

import torch
from torch.utils.data import Dataset

BASE = "Qwen/Qwen2.5-7B-Instruct"
MAX_LEN = 2048
random.seed(42)


def load_rows(p, upsample_pos=1):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    out = []
    for r in rows:
        out.append(r)
        if r["label"] == "支持":
            out.extend([r] * (upsample_pos - 1))
    random.shuffle(out)
    return out


class VerifierDS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # 注意：transformers v5 的 apply_chat_template(tokenize=True) 返回dict，
        # 这里显式走 文本->tokenize 两步
        prompt_text = self.tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        label_ids = self.tok(r["label"], add_special_tokens=False)["input_ids"]
        label_ids = label_ids + [self.tok.eos_token_id]
        # 截断上下文中段，保留题目和判定要求（prompt尾部）
        if len(prompt_ids) + len(label_ids) > MAX_LEN:
            keep = MAX_LEN - len(label_ids)
            prompt_ids = prompt_ids[:keep // 2] + prompt_ids[-(keep - keep // 2):]
        ids = prompt_ids + label_ids
        labels = [-100] * len(prompt_ids) + label_ids
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        d = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * d)
        labels.append(b["labels"] + [-100] * d)
        mask.append([1] * (n - d) + [0] * d)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_file")
    ap.add_argument("val_file")
    ap.add_argument("-o", "--out", default="models/verifier_lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True),
        device_map={"": 0}, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.config.use_cache = False

    train_ds = VerifierDS(load_rows(args.train_file, upsample_pos=3), tok)
    val_ds = VerifierDS(load_rows(args.val_file), tok)
    print(f"train {len(train_ds)} / val {len(val_ds)}")

    targs = TrainingArguments(
        output_dir=args.out + "_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=1,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=lambda b: collate(b, tok.pad_token_id))
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"LoRA adapter 已保存到 {args.out}")


if __name__ == "__main__":
    main()
