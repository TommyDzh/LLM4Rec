#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline:
1) Load metadata and assemble item texts
2) Encode texts with BGE-M3 to dense embeddings
3) Train Faiss KMeans and assign clusters
4) Export per-cluster top-k nearest items (for naming)
5) (Optional) Name clusters via LLM (Ark API)
6) Load user history, aggregate per-cluster pos/neg counts
7) Build user feature prompts and save to JSONL
"""

import os
import re
import json
import time
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from FlagEmbedding import BGEM3FlagModel
import faiss


# ------------------------------
# Utilities
# ------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def detect_id_column(df_meta: pd.DataFrame) -> str:
    """Detect an item id column in meta."""
    for cand in ["parent_asin", "asin", "item_id", "id"]:
        if cand in df_meta.columns:
            return cand
    raise KeyError("No item id column found in meta. Expected one of: parent_asin, asin, item_id, id.")


def join_categories(cats) -> str:
    """Join hierarchical categories into a single string with '|'."""
    try:
        if len(cats) > 0 and isinstance(cats[0], list):
            flat = sum(cats, [])
        else:
            flat = list(cats)
        return "|".join([str(x) for x in flat])
    except Exception:
        return ""


def join_features(feats) -> str:
    """Join features into a single string with '###'."""
    try:
        return "###".join([str(x) for x in feats])
    except Exception:
        return ""


# ------------------------------
# Loading / Text building
# ------------------------------

def load_meta(meta_path: str) -> pd.DataFrame:
    return pd.read_json(meta_path, lines=True)


def build_item_texts(df_meta: pd.DataFrame, id_col: str) -> list[str]:
    texts = []
    for row in df_meta.itertuples(index=False):
        title = getattr(row, "title", "")
        categories = getattr(row, "categories", [])
        features = getattr(row, "features", [])
        cat_str = join_categories(categories)
        feat_str = join_features(features)
        texts.append(f"Title: {title}.\nCategory: {cat_str}.\nFeature: {feat_str}.")
    return texts


# ------------------------------
# Embedding with BGE-M3
# ------------------------------

def encode_bge_m3(
    texts: list[str],
    model_id: str = "BAAI/bge-m3",
    batch_size: int = 128,
    max_length: int = 1024,
    use_fp16: bool = True,
    device: str = "auto",
) -> np.ndarray:
    model = BGEM3FlagModel(model_id, use_fp16=use_fp16, device=device)
    outs = model.encode(texts, batch_size=batch_size, max_length=max_length)
    dense = outs["dense_vecs"].astype(np.float32)
    return dense


# ------------------------------
# KMeans with Faiss
# ------------------------------

def train_faiss_kmeans(
    embeddings: np.ndarray,
    n_clusters: int = 20,
    niter: int = 30,
    nredo: int = 5,
    use_gpu: bool = True,
    sample_ratio: float = 1.0,
):
    d = embeddings.shape[1]
    kwargs = dict(d=d, k=n_clusters, niter=niter, nredo=nredo, verbose=True)
    # max_points_per_centroid helps speed on large corpora
    kwargs["max_points_per_centroid"] = max(1, int(len(embeddings) // max(1, n_clusters) * sample_ratio))

    print(f"[KMeans] d={d}, k={n_clusters}, niter={niter}, nredo={nredo}, gpu={use_gpu}")
    try:
        kmeans = faiss.Kmeans(**kwargs, gpu=use_gpu)  # some builds support gpu=...
    except TypeError:
        # fallback: no gpu arg in this faiss build
        kmeans = faiss.Kmeans(**kwargs)

    t0 = time.time()
    kmeans.train(embeddings)
    t1 = time.time()
    print(f"[KMeans] Training finished in {t1 - t0:.3f}s")

    # assign clusters
    D, I = kmeans.index.search(embeddings, 1)
    print(f"[KMeans] Final objective: {kmeans.obj[-1]:.4f}")

    return kmeans.centroids, D.flatten(), I.flatten(), kmeans.obj[-1]


def top_k_nearest_samples(cluster_ids: np.ndarray, distances: np.ndarray, top_k: int = 100) -> dict[int, list[int]]:
    """
    Returns top-k closest indices per cluster.
    `distances` are per-item distances to its assigned centroid.
    """
    cluster_to_indices = {}
    n_clusters = int(cluster_ids.max()) + 1

    for cid in range(n_clusters):
        mask = (cluster_ids == cid)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            cluster_to_indices[cid] = []
            continue
        dists = distances[mask]
        local_sorted = dists.argsort()[:top_k]
        cluster_to_indices[cid] = idxs[local_sorted].tolist()

    return cluster_to_indices


# ------------------------------
# Optional: LLM naming (Ark)
# ------------------------------

def name_clusters_with_ark(
    cluster_search: dict[int, list[str]],
    ark_base_url: str,
    ark_model: str,
    category_name: str,
    api_key_env: str = "ARK_API_KEY",
) -> list[str]:
    """
    Requires `volcenginesdkarkruntime` installed and env var ARK_API_KEY set.
    Returns a list of raw LLM responses (one per cluster).
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"Env var {api_key_env} not set. Export your Ark API key first.")

    try:
        from volcenginesdkarkruntime import Ark
    except Exception as e:
        raise RuntimeError("volcenginesdkarkruntime is not installed. `pip install volcenginesdkarkruntime`") from e

    client = Ark(base_url=ark_base_url, api_key=api_key)
    cluster_topics = []

    for cid in range(len(cluster_search)):
        center_searches = "\n".join(cluster_search[cid])
        prompt = f"""
The following are 100 {category_name} that have been purchased by the same type of users on Amazon, indicating they belong to the same thematic category. Each {category_name} entry includes its title, category tags, and a brief description. Please analyze the implicit commonalities among these {category_name} and infer their shared theme.

Please summarize based on the following aspects:
- What are the common interests or needs of the users?
- What types of keywords, topics, or genres appear frequently?
- What core ideas or values do these {category_name} aim to convey or explore?

[Constraints]:
- Summarize the theme in one short phrase (<= 10 words).
- Use concise, abstract language; avoid sentences.
- Do not list specific {category_name} titles or examples.

Below is the list of {category_name} (each includes title, categories, brief intro):
[{center_searches}]

Please output:
- Theme: [Concise phrase, e.g., "Political conspiracy & whistleblower thrillers"]
"""
        resp = client.chat.completions.create(
            model=ark_model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            extra_headers={"x-is-encrypted": "true"},
        )
        text = resp.choices[0].message.content
        print(f"[Ark] Cluster {cid}: {text}")
        cluster_topics.append(text)

    return cluster_topics


def extract_theme_line(raw: str) -> str:
    """Extract 'Theme: ...' from LLM output."""
    # Try common formats
    m = re.search(r"[Tt]heme\s*:\s*(.+)", raw)
    if m:
        x = m.group(1).strip()
        # drop trailing decorations or following lines
        return x.splitlines()[0].strip().strip("*")
    return raw.strip()


# ------------------------------
# User history -> per-cluster features
# ------------------------------

def build_asin_to_cluster(df_meta: pd.DataFrame, id_col: str, cluster_ids: np.ndarray) -> dict:
    """Map item id to its cluster id. Order of cluster_ids == order in df_meta."""
    if len(df_meta) != len(cluster_ids):
        raise ValueError("Meta length and cluster_ids length mismatch.")
    asin2cluster = {}
    for i, row in enumerate(df_meta[id_col].tolist()):
        asin2cluster[row] = int(cluster_ids[i])
    return asin2cluster


def compute_user_cluster_counts(
    df_history: pd.DataFrame,
    asin2cluster: dict,
    n_clusters: int,
    pos_threshold: int = 4,
) -> tuple[dict, dict]:
    """
    df_history is expected to have columns: user_id, history_asins, history_ratings
    Returns: user_pos_count[uid] -> np.array[K], user_neg_count[uid] -> np.array[K]
    """
    user_pos = defaultdict(lambda: np.zeros(n_clusters, dtype=np.int32))
    user_neg = defaultdict(lambda: np.zeros(n_clusters, dtype=np.int32))

    for row in df_history.itertuples(index=False):
        uid = getattr(row, "user_id")
        asins = getattr(row, "history_asins")
        ratings = getattr(row, "history_ratings")
        if asins is None or ratings is None:
            continue
        for a, r in zip(asins, ratings):
            cid = asin2cluster.get(a)
            if cid is None:
                continue
            if int(r) >= pos_threshold:
                user_pos[uid][cid] += 1
            else:
                user_neg[uid][cid] += 1
    return user_pos, user_neg


def build_user_feature_prompt(row: pd.Series) -> str:
    """Construct a two-list prompt from per-cluster (pos_count, neg_count) tuples."""
    parts_pos, parts_neg = [], []
    for col in row.index:
        if col in ["user_id", "user_feature_prompt"]:
            continue
        val = row[col]
        if isinstance(val, (tuple, list)) and len(val) == 2:
            pos, neg = val
            if (pos + neg) > 0:
                if pos > neg:
                    parts_pos.append(f"- {col}")
                elif neg > pos:
                    parts_neg.append(f"- {col}")

    pos_prompt, neg_prompt = "", ""
    if parts_pos:
        pos_prompt = "The user shows preference for the following topics (ratings ≥ 4):\n" + "\n".join(parts_pos)
    if parts_neg:
        neg_prompt = "The user shows dislike for the following topics (ratings ≤ 3):\n" + "\n".join(parts_neg)

    return "\n\n".join([p for p in [pos_prompt, neg_prompt] if p])


# ------------------------------
# Main
# ------------------------------

def main(args: argparse.Namespace):
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join("data", args.data_category))

    # 1) Load meta and build texts
    print("[Load] meta:", args.meta_path)
    df_meta = load_meta(args.meta_path)
    id_col = detect_id_column(df_meta)
    texts = build_item_texts(df_meta, id_col=id_col)
    print(f"[Load] items: {len(texts)}")

    # 2) Encode
    emb = encode_bge_m3(
        texts,
        model_id=args.encoder_model,
        batch_size=args.encode_batch_size,
        max_length=args.encode_max_len,
        use_fp16=not args.no_fp16,
        device=args.device,
    )
    print(f"[Encode] embeddings: {emb.shape}")

    # 3) KMeans
    _, distances, cluster_ids, obj = train_faiss_kmeans(
        emb,
        n_clusters=args.n_clusters,
        niter=args.kmeans_niter,
        nredo=args.kmeans_nredo,
        use_gpu=not args.no_kmeans_gpu,
        sample_ratio=args.kmeans_sample_ratio,
    )

    # Save KMeans assignments
    kmeans_npz = os.path.join(args.output_dir, f"kmeans_results_{args.n_clusters}.npz")
    np.savez_compressed(kmeans_npz, distances=distances, cluster_ids=cluster_ids)
    print(f"[Save] {kmeans_npz}")

    # 4) Per-cluster top-k nearest items
    cluster_to_top = top_k_nearest_samples(cluster_ids, distances, top_k=args.top_k)
    cluster_search = {cid: [texts[i] for i in idxs] for cid, idxs in cluster_to_top.items()}
    cluster_json = os.path.join(args.output_dir, f"cluster_{args.data_category}_{args.encoder_tag}_{args.n_clusters}.json")
    with open(cluster_json, "w") as f:
        json.dump(cluster_search, f, ensure_ascii=False, indent=2)
    print(f"[Save] {cluster_json}")

    # 5) (Optional) Name clusters with LLM
    if args.name_clusters:
        raw_topics = name_clusters_with_ark(
            cluster_search=cluster_search,
            ark_base_url=args.ark_base_url,
            ark_model=args.ark_model,
            category_name=args.category_name,
            api_key_env=args.ark_api_key_env,
        )
        topics = [extract_theme_line(t) for t in raw_topics]

        # disambiguate duplicate topic names by suffix
        counts = Counter(topics)
        idx_map = defaultdict(int)
        final_topics = []
        for t in topics:
            if counts[t] > 1:
                j = idx_map[t]
                final_topics.append(f"{t}_{j}")
                idx_map[t] += 1
            else:
                final_topics.append(t)

        topics_path = os.path.join(
            args.output_dir,
            f"cluster_topics_{args.encoder_tag}_{args.data_category}_{args.topic_days}_{args.n_clusters}-full.json"
        )
        with open(topics_path, "w") as f:
            json.dump(raw_topics, f, ensure_ascii=False, indent=2)
        print(f"[Save] raw topics -> {topics_path}")

        # Use final_topics as ordered column names later
    else:
        # If no naming, use "Topic_i"
        final_topics = [f"Topic_{i}" for i in range(args.n_clusters)]
        topics_path = os.path.join(
            args.output_dir,
            f"cluster_topics_{args.encoder_tag}_{args.data_category}_{args.topic_days}_{args.n_clusters}-auto.json"
        )
        with open(topics_path, "w") as f:
            json.dump(final_topics, f, ensure_ascii=False, indent=2)
        print(f"[Save] auto topics -> {topics_path}")

    # 6) User history aggregation
    if not args.history_path:
        raise ValueError("--history_path is required for building user feature prompts.")

    df_history = pd.read_json(args.history_path, lines=True)
    print(f"[Load] history: {len(df_history)}")

    asin2cluster = build_asin_to_cluster(df_meta, id_col=id_col, cluster_ids=cluster_ids)
    user_pos, user_neg = compute_user_cluster_counts(
        df_history=df_history,
        asin2cluster=asin2cluster,
        n_clusters=args.n_clusters,
        pos_threshold=args.pos_threshold,
    )

    # 7) Build user feature frame
    # Columns = topic names; cell = (pos_count, neg_count)
    user_feature_list = {}
    for uid in user_pos:
        pos_arr = user_pos[uid].tolist()
        neg_arr = user_neg[uid].tolist()
        user_feature_list[uid] = [(p, n) for p, n in zip(pos_arr, neg_arr)]

    df_user = pd.DataFrame.from_dict(user_feature_list, orient="index", columns=final_topics)
    df_user["user_id"] = df_user.index
    df_user = df_user[["user_id"] + final_topics]

    # Build prompts
    df_user["user_feature_prompt"] = df_user.apply(build_user_feature_prompt, axis=1)

    # Save prompts
    out_jsonl = os.path.join(
        args.output_dir,
        f"user_two_feature_{args.n_clusters}_list.jsonl"
    )
    pd.DataFrame(
        list(zip(df_user["user_id"], df_user["user_feature_prompt"])),
        columns=["user_id", "user_profile_prompt"],
    ).to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    print(f"[Save] {out_jsonl}")

    print("[Done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interest sketch clustering and user feature prompt builder.")

    # Core I/O
    parser.add_argument("--data_category", type=str, default="Books", help="Category name (e.g., Books, Movies).")
    parser.add_argument("--meta_path", type=str,
                        default="/mnt/hdfs/shark/zhihao/AmazonReviews/Books/meta_Books.jsonl.gz",
                        help="Path to item meta JSONL.gz.")
    parser.add_argument("--history_path", type=str, required=True,
                        help="Path to user history JSONL.gz with columns: user_id, history_asins, history_ratings.")
    parser.add_argument("--output_dir", type=str, default="data/Books",
                        help="Directory to store outputs (npz/json/jsonl).")

    # Embedding
    parser.add_argument("--encoder_model", type=str, default="BAAI/bge-m3", help="Embedding model id.")
    parser.add_argument("--encode_batch_size", type=int, default=128)
    parser.add_argument("--encode_max_len", type=int, default=1024)
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 for embedding.")
    parser.add_argument("--device", type=str, default="auto", help="Device for embedding model.")

    # KMeans
    parser.add_argument("--n_clusters", type=int, default=20)
    parser.add_argument("--kmeans_niter", type=int, default=30)
    parser.add_argument("--kmeans_nredo", type=int, default=5)
    parser.add_argument("--kmeans_sample_ratio", type=float, default=1.0)
    parser.add_argument("--no_kmeans_gpu", action="store_true", help="Force CPU KMeans if set.")

    # Top-k selection
    parser.add_argument("--top_k", type=int, default=100, help="Top-k nearest items per cluster for naming.")

    # LLM naming (optional)
    parser.add_argument("--name_clusters", action="store_true", help="Enable LLM-based cluster naming.")
    parser.add_argument("--ark_base_url", type=str, default="https://ark-cn-beijing.bytedance.net/api/v3")
    parser.add_argument("--ark_model", type=str, default="ep-20250205142933-sld5j")
    parser.add_argument("--ark_api_key_env", type=str, default="ARK_API_KEY",
                        help="Env var name that stores Ark API key.")
    parser.add_argument("--category_name", type=str, default="book",
                        help="Human-readable item type for prompts (e.g., 'book', 'movie').")
    parser.add_argument("--topic_days", type=int, default=999, help="Tag to mark topic naming batch.")

    # User preference logic
    parser.add_argument("--pos_threshold", type=int, default=4, help="Rating >= threshold => positive.")

    # Encoder tag for filenames
    parser.add_argument("--encoder_tag", type=str, default="BGE", help="Short tag for encoder id to embed in filenames.")

    args = parser.parse_args()
    main(args)
