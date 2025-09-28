import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset,Dataset,load_from_disk
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
# from accelerate.utils import load_balanced_sampler
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

import re
import numpy as np
from glob import glob
def extract_answer(text):
    # 使用非贪婪匹配模式提取<ANSWER>标签之间的内容
    pattern = r'<ANSWER>(.*?)</ANSWER>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0]
# def collate_fn(batch):
#     prompts, user_ids = zip(*batch)
#     return list(prompts), list(user_ids)
class InferenceDataset(Dataset):
    def __init__(self, prompts, user_ids):
        self.prompts = prompts
        self.user_ids = user_ids

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # if isinstance(idx, list) or isinstance(idx, np.ndarray):
        #     return [self.prompts[i] for i in idx], [self.user_ids[i] for i in idx]
        return self.prompts[idx], self.user_ids[idx]


def main(args):
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    dataset_train = pd.read_parquet(os.path.join(args.data_path, "train.parquet"))
    with open(os.path.join(args.thinking_path, "results_thinking.json"), "r") as f:
        results_thinking = json.load(f)

    full_text = {}
    for test_id, (test_id, row) in enumerate(dataset.iterrows()):
        text = tokenizer.apply_chat_template(row.prompt, tokenize=False, add_generation_prompt=True)
        thinking = results_thinking[str(test_id)]
        new_text = text + "\n" + thinking + "\n" + "<answer>"
        full_text[test_id] = new_text



    model = AutoModelForCausalLM.from_pretrained(args.model,  torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
    model.eval()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    prompts, user_ids= [], []
    
    for test_id, text in full_text.items():
        test_id = str(test_id)

        
        prompts.append(text)
        user_ids.append(test_id)


    dataset = InferenceDataset(prompts, user_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataloader = accelerator.prepare(dataloader)



    hidden_dict = {}
    num = 0
    for batch in tqdm(dataloader):
        batch_prompts, batch_user_ids = batch
        model_inputs = tokenizer(list(batch_prompts), return_tensors="pt", padding="longest", truncation=False, max_length=3000).to(accelerator.device)

        with torch.no_grad():
            outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        for idx, user_id in enumerate(batch_user_ids):
            user_hiddens = []
            for h in hidden_states:
                user_hiddens.append(h[idx, -1, :].cpu())
            user_hiddens = torch.vstack(user_hiddens)
            hidden_dict[user_id] = user_hiddens
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    torch.save(hidden_dict,  os.path.join(args.out_dir, f"last_hidden_dict_{accelerator.process_index}.pt"))


    

    accelerator.wait_for_everyone() 

    if accelerator.is_main_process:
        os.makedirs(args.out_dir, exist_ok=True)
        shard_paths = sorted(
            glob(os.path.join(args.out_dir, "last_hidden_dict_*.pt")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[-1])
        )

        merged = {}
        dup_keys = []

        for p in shard_paths:
            part = torch.load(p, map_location="cpu")
            # 合并：若 key 冲突，保留第一次并记录冲突
            for k, v in part.items():
                if k in merged:
                    dup_keys.append(k)
                    # 如果你想覆盖而非跳过，可以改成：merged[k] = v
                else:
                    merged[k] = v
        hidden_dict = {}
        for i in range(len(merged)):
            hidden_dict[str(i)] = merged[str(i)]
        out_merged = os.path.join(args.out_dir, "last_hidden_dict.pt")
        torch.save(hidden_dict, out_merged)
        print(f"[Main] Merged {len(shard_paths)} shards -> {out_merged} with {len(merged)} keys.")

        if dup_keys:
            print(f"[Main][Warn] {len(dup_keys)} duplicated keys encountered. First occurrence kept. "
                f"Example: {dup_keys[:5]}")

    accelerator.wait_for_everyone()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ckpt/GRPO_max@10-long@False_step420")
    parser.add_argument("--thinking_path", type=str, default="results_train/Movies/GRPO_max@10-long@False_step420", help="Path to training jsonl")
    parser.add_argument("--data_path", type=str, default="data/Movies/verl", help="Path to training jsonl")
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--out_dir", type=str, default="results_train/Movies", help="Path to training jsonl")
    parser.add_argument("--thinking", type=bool, default=False)
    args = parser.parse_args()
    main(args)
     