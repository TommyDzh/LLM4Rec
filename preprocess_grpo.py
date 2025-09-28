#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Preprocess AmazonReviews to GRPO-style parquet with prompts & rewards.

Usage (example):
  python preprocess.py \
    --category Movies \
    --data_root /mnt/hdfs/zhihao/AmazonReviews \
    --max_history 10 \
    --use_long_history False \
    --out_dir data_movie/Movies/verl/max@10-long@False

Notes:
  - Requires: pandas, pyarrow
    pip install pandas pyarrow
"""

import os
import argparse
import pandas as pd


SYSTEM_PROMPT_TPL = (
    "You are an intelligent recommender system assistant. "
    "Your task is to predict the rating (from 1.0 to 5.0) that a user will give "
    "to a candidate {category}, based on the user's past behavior and preferences.\n\n"
    "You should first reason about how well the candidate item aligns with the user’s historical preferences, "
    "and then output a predicted rating. Your reasoning and the final rating must be wrapped with "
    "<think> </think> and <answer> </answer> tags, respectively.\n\n"
    "The final rating **must be a numeric value between 1.0 and 5.0**. "
    "Do not include any extra explanation after the <answer> tag."
)


def generate_feature_rating_prompt_grpo(
    target: str,
    short_history: str,
    long_history: str | None = None,
    history_feature: str | None = "",
    category_lower: str = "movie",
    max_history: int = 10,
) -> str:
    """Compose the user prompt with (optional) long history, short history, and user profile feature."""
    intro = f"""{history_feature or ""}

Below are the user's recent {category_lower} ratings in the format: Title, Genres, Rating. Ratings range from 1.0 to 5.0.
""".lstrip()

    # 拼接历史
    history_concat = (long_history + "\n" + short_history) if long_history else short_history
    lines = [ln for ln in (history_concat or "").split("\n") if ln.strip()]
    history_tail = "\n".join(lines[-max_history:])

    tail = (
        f"\n{history_tail}\n\n"
        f"The candidate {category_lower} is described as: {target}\n"
        f"Based on the above information, what rating will the user give?"
    )
    return intro + tail


def make_row_processor(
    category: str,
    category_lower: str,
    max_history: int,
    use_long_history: bool,
    system_prompt: str,
):
    """Closure to produce a row->Series converter with bound args."""
    data_source = f"AmazonReviews/{category}"

    def process_sample(row: pd.Series) -> pd.Series:
        # 原始字段健壮提取
        history = str(row.get("history", "") or "")
        target = str(row.get("target", "") or "")
        label = row.get("label", None)
        long_history = str(row.get("long_history", "") or "") if use_long_history else None
        history_feature = str(row.get("user_profile_prompt", "") or "")

        # 拼 prompt
        question = generate_feature_rating_prompt_grpo(
            target=target,
            short_history=history,
            long_history=long_history,
            history_feature=history_feature,
            category_lower=category_lower,
            max_history=max_history,
        )

        user_id = row.get("user_id", None)
        parent_asin = row.get("parent_asin", None)
        split = row.get("split", None)

        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "ability": "logic",
            "reward_model": {
                "style": "rule",
                "ground_truth": str(label) if label is not None else None,
            },
            "extra_info": {
                "split": split,
                "user_id": user_id,
                "parent_asin": parent_asin,
            },
        }
        return pd.Series(data)

    return process_sample


def load_split_jsonl_gz(path: str, split_name: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["split"] = split_name
    return df


def main(args: argparse.Namespace) -> None:
    category = args.category  # "Books" or "Movies"
    category_lower = category.lower().rstrip("s")  # "book"/"movie"
    data_root = args.data_root.rstrip("/")

    split_dir = os.path.join(data_root, category)
    p_train = os.path.join(split_dir, "train.jsonl.gz")
    p_valid = os.path.join(split_dir, "valid.jsonl.gz")
    p_test  = os.path.join(split_dir, "test.jsonl.gz")

    df_train = load_split_jsonl_gz(p_train, "train")
    df_valid = load_split_jsonl_gz(p_valid, "valid")
    df_test  = load_split_jsonl_gz(p_test,  "test")
    print(f"[Info] Loaded: train={len(df_train)}, valid={len(df_valid)}, test={len(df_test)}")

    feat_name = f"user_feature.jsonl"
    p_feat = os.path.join(split_dir, feat_name)
    if os.path.exists(p_feat):
        df_feature = pd.read_json(p_feat, lines=True)
        print(f"[Info] Loaded feature: {feat_name} (rows={len(df_feature)})")
        df_train = df_train.merge(df_feature, on="user_id", how="left")
        df_valid = df_valid.merge(df_feature, on="user_id", how="left")
        df_test  = df_test.merge(df_feature, on="user_id", how="left")
    else:
        print(f"[Warn] Feature file not found: {p_feat}. Proceeding without user_profile_prompt.")

    # system prompt
    system_prompt = SYSTEM_PROMPT_TPL.format(category=category_lower)


    row_processor = make_row_processor(
        category=category,
        category_lower=category_lower,
        max_history=args.max_history,
        use_long_history=args.use_long_history,
        system_prompt=system_prompt,
    )


    out_dir = args.out_dir
    if not out_dir:
        long_flag = "True" if args.use_long_history else "False"
        out_dir = os.path.join(
            f"data_{category_lower}",
            category,
            "verl",
            f"max@{args.max_history}-long@{long_flag}",
        )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Info] Output dir: {out_dir}")

    dataset_train = df_train.apply(row_processor, axis=1)
    dataset_val   = df_valid.apply(row_processor, axis=1)
    dataset_test  = df_test.apply(row_processor, axis=1)


    f_train = os.path.join(out_dir, "train.parquet")
    f_val   = os.path.join(out_dir, "val.parquet")
    f_test  = os.path.join(out_dir, "test.parquet")

    dataset_train.to_parquet(f_train, index=False)
    dataset_val.to_parquet(f_val, index=False)
    dataset_test.to_parquet(f_test, index=False)

    print(f"[Done] Wrote: {f_train}")
    print(f"[Done] Wrote: {f_val}")
    print(f"[Done] Wrote: {f_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default="Movies", choices=["Books", "Movies"],
                        help="Dataset category.")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root folder that contains <category> subfolder.")
    parser.add_argument("--n_clusters", type=int, default=10, help="Cluster count used in feature filename.")
    parser.add_argument("--max_history", type=int, default=10, help="Max history entries to keep in prompt.")
    parser.add_argument("--use_long_history", type=bool, default=False,
                        help="Whether to include `long_history` field if available.")
    parser.add_argument("--out_dir", type=str, default="",
                        help="Output directory. If empty, a default path will be constructed.")
    args = parser.parse_args()
    main(args)
