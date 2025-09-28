#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def ensure_chat_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    raise ValueError("`prompt` must be a list[{'role','content'}] or its JSON string.")


def extract_ground_truth(obj):
    if isinstance(obj, dict) and "ground_truth" in obj:
        val = obj["ground_truth"]
        return "" if val is None else str(val)
    if obj is None:
        return ""
    return str(obj)


def build_prompt_completion(df, tokenizer, suffix="\n<think>\n</think>\n<answer>"):
    data = []
    for row in df.itertuples(index=False):
        chat = ensure_chat_list(getattr(row, "prompt"))
        completion = extract_ground_truth(getattr(row, "reward_model", None))
        if not completion and hasattr(row, "ground_truth"):
            completion = str(getattr(row, "ground_truth"))
        rendered = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        rendered += suffix
        data.append({"prompt": rendered, "completion": completion})
    return Dataset.from_list(data)


def read_split(parquet_dir, split, prefer_train_new=False):
    paths = []
    if split == "train" and prefer_train_new:
        paths.append(os.path.join(parquet_dir, "train_new.parquet"))
    paths.append(os.path.join(parquet_dir, f"{split}.parquet"))
    for p in paths:
        if os.path.exists(p):
            return pd.read_parquet(p)
    raise FileNotFoundError(f"No parquet found for split {split} in {paths}")


def main(args):
    data_name = f"max@{args.max_history}-long@{args.use_long_history}"
    parquet_dir = os.path.join(args.data_base, args.category, "verl", data_name)
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    df_train = read_split(parquet_dir, "train", False)
    df_valid = read_split(parquet_dir, "val", False)
    df_test = read_split(parquet_dir, "test", False)

    ds_train = build_prompt_completion(df_train, tokenizer)
    ds_valid = build_prompt_completion(df_valid, tokenizer)
    ds_test = build_prompt_completion(df_test, tokenizer)

    dataset = DatasetDict({"train": ds_train, "valid": ds_valid, "test": ds_test})

    out_dir = args.out_dir or os.path.join(args.data_base, data_name)
    os.makedirs(out_dir, exist_ok=True)
    dataset.save_to_disk(out_dir)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="Qwen3-4B")
    parser.add_argument("--category", type=str, default="Movies", choices=["Books", "Movies"])
    parser.add_argument("--data_base", type=str, default="data")
    parser.add_argument("--max_history", type=int, default=10)
    parser.add_argument("--use_long_history", type=bool, default=False)
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()
    main(args)
