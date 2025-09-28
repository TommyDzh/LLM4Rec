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
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnAnswerTag(StoppingCriteria):
    def __init__(self, tokenizer, tag="<|endoftext|>"):
        self.tag_ids = tokenizer(tag, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        self.len = len(self.tag_ids)
    def __call__(self, input_ids, scores, **kwargs):
        seq = input_ids[0].tolist() if input_ids.dim()==2 else input_ids.tolist()
        # check suffix match (cheap)
        return len(seq) >= self.len and seq[-self.len:] == self.tag_ids


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
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    dataset_name = args.data_path.split("/")[-1].split(".")[0]
    dataset = pd.read_parquet(os.path.join(args.data_path, "train_new.parquet"))





    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    stopper = StoppingCriteriaList([StopOnAnswerTag(tokenizer, "<|endoftext|>")])
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
    model.eval()
    model, tokenizer = accelerator.prepare(model, tokenizer)
    # model.to(accelerator.device)
    # tokenizer.to(accelerator.device)  # tokenizer 通常不需要 to(device)，除非特殊模型结构


    prompts, user_ids= [], []
    label_dict = {}
    
    for test_id, row in dataset.iterrows():
        test_id = str(test_id)
        text = tokenizer.apply_chat_template(row.prompt, tokenize=False, add_generation_prompt=True)
        label = float(row.reward_model["ground_truth"])
        
        prompts.append(text)
        user_ids.append(test_id)
        label_dict[test_id] = label


    dataset = InferenceDataset(prompts, user_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    dataloader = accelerator.prepare(dataloader)



    results_dict = {}
    results_thinking_dict = {}
    logits_dict = {}
    full_dict = {}
    num = 0
    for batch in tqdm(dataloader):
        batch_prompts, batch_user_ids = batch
        model_inputs = tokenizer(list(batch_prompts), return_tensors="pt", padding="longest", truncation=False).to(accelerator.device)

        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(
                **model_inputs,
                max_new_tokens=1024,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                use_cache=True,
                # stopping_criteria=stopper
            )
        for j, generated in enumerate(outputs.sequences):
            input_len = len(model_inputs["input_ids"][j])
            output_ids = generated[input_len:].tolist()
            uid =batch_user_ids[j]
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
            except ValueError:
                index = 0
            thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            result = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip().upper()
            full_text = tokenizer.decode(output_ids, skip_special_tokens=False)
            try:
                result = float(extract_answer(result.upper()))
            except:
                print(result)
                result = "0.0"
            results_thinking_dict[str(uid)] = thinking
            results_dict[str(uid)] = result
            full_dict[str(uid)] = full_text
            num +=1
            if num == 1:
                print("###Prompt:", batch_prompts[0])
                print("###think:", thinking)
                print("###result:", result)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    results_list = [{"user_id": str(uid), "result": result} for uid, result in results_dict.items()]
    thinking_list = [{"user_id": str(uid), "thinking": results_thinking_dict[uid]} for uid in results_thinking_dict]
    full_list = [{"user_id": str(uid), "full_text": full_dict[uid]} for uid in full_dict]

    gathered_results = accelerator.gather_for_metrics(results_list)
    gathered_thinking = accelerator.gather_for_metrics(thinking_list)
    gathered_full = accelerator.gather_for_metrics(full_list)
    if accelerator.is_main_process:
        results_dict = {item["user_id"]: item["result"] for item in gathered_results}
        results_thinking_dict = {item["user_id"]: item["thinking"] for item in gathered_thinking}
        full_dict = {item["user_id"]: item["full_text"] for item in gathered_full}
        os.makedirs(args.out_dir, exist_ok=True)

        output_dir = os.path.join(args.out_dir, f"{model_name}")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"results.json"), "w") as f:
            json.dump(results_dict, f)

        with open(os.path.join(output_dir, f"results_thinking.json"), "w") as f:
            json.dump(results_thinking_dict, f)
        with open(os.path.join(output_dir, f"full_text.json"), "w") as f:
            json.dump(full_dict, f)
        # Evaluate
        invalid = 0
        labels = []
        preds = []
        mask = []
        acc = 0
        for tid, pred in results_dict.items():
            label = label_dict[tid]
            labels.append(label)
            try:
                pred = float(pred)
            except:
                print(pred)
                pred = 5.0
                invalid += 1
                preds.append(pred)
                mask.append(0)
                continue
            preds.append(pred)
            
            if pred == label:
                acc += 1
                mask.append(1)
            else:
                mask.append(0)
        print(f"Invalid: {invalid}/{len(results_dict)}", invalid/len(results_dict))
        mae = mean_absolute_error(labels, preds)
        rmse = root_mean_squared_error(labels, preds) 
        print(f"MAE: {mae:.4f}; RMSE: {rmse:.4f}; Acc: {acc/len(results_dict):.4f}")
        torch.save(mask, os.path.join(output_dir, f"mask.pt"))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ckpt/GRPO_max@10-long@False_step420")
    parser.add_argument("--data_path", type=str, default="data/Movies/verl/max@10-long@False", help="Path to training jsonl")
    parser.add_argument("--out_dir", type=str, default="results_train/Movies", help="Path to training jsonl")
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--thinking", type=bool, default=False)
    args = parser.parse_args()
    main(args)
     