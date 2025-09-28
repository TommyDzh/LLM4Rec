import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from accelerate import Accelerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
# from accelerate.utils import load_balanced_sampler
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datasets import load_dataset,Dataset,load_from_disk
from torch.utils.data import Dataset, DataLoader
import re

def extract_answer(text):
    pattern = r'<ANSWER>(.*?)</ANSWER>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0]

class InferenceDataset(Dataset):
    def __init__(self, prompts, user_ids):
        self.prompts = prompts
        self.user_ids = user_ids

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.user_ids[idx]


def main(args):
    accelerator = Accelerator()
    model_name = args.model.split("/")[-1] if "/" in args.model else args.model
    dataset_name = args.data_path.split("/")[-1].split(".")[0]
    dataset = load_from_disk(args.data_path)
    dataset_test = dataset["test"]




    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    model.eval()
    model, tokenizer = accelerator.prepare(model, tokenizer)


    prompts, user_ids= [], []
    label_dict = {}
    
    for test_id, row in enumerate(dataset_test):
        text = row["prompt"]
        label = float(row["completion"])
        
        prompts.append(text)
        user_ids.append(test_id)
        label_dict[str(test_id)] = label


    dataset = InferenceDataset(prompts, user_ids)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    dataloader = accelerator.prepare(dataloader)



    results_dict = {}
    results_thinking_dict = {}
    logits_dict = {}
    num = 0
    for batch in tqdm(dataloader):
        batch_prompts, batch_user_ids = batch
        model_inputs = tokenizer(list(batch_prompts), return_tensors="pt", padding="longest", truncation=True).to(accelerator.device)

        with torch.no_grad():
            outputs = accelerator.unwrap_model(model).generate(
                **model_inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
            )
        for j, generated in enumerate(outputs.sequences):
            input_len = len(model_inputs["input_ids"][j])
            output_ids = generated[input_len:].tolist()
            uid =batch_user_ids[j].item()
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
            except ValueError:
                index = 0
            thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            result = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip().upper()
            try:
                result = float(result)
            except:
                print(result)
                result = "5.0"
            results_thinking_dict[str(uid)] = thinking
            results_dict[str(uid)] = result
            num +=1
            if num == 1:
                print("###Prompt:", batch_prompts[0])
                print("###think:", thinking)
                print("###result:", result)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
    results_list = [{"user_id": str(uid), "result": result} for uid, result in results_dict.items()]
    thinking_list = [{"user_id": str(uid), "thinking": results_thinking_dict[uid]} for uid in results_thinking_dict]

    gathered_results = accelerator.gather_for_metrics(results_list)
    gathered_thinking = accelerator.gather_for_metrics(thinking_list)
    if accelerator.is_main_process:
        results_dict = {item["user_id"]: item["result"] for item in gathered_results}
        results_thinking_dict = {item["user_id"]: item["thinking"] for item in gathered_thinking}
        os.makedirs(args.out_dir, exist_ok=True)

        output_dir = os.path.join(args.out_dir, f"{model_name}")
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"results.json"), "w") as f:
            json.dump(results_dict, f)

        with open(os.path.join(output_dir, f"results_thinking.json"), "w") as f:
            json.dump(results_thinking_dict, f)
        # Evaluate
        invalid = 0
        labels = []
        preds = []
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
        print(f"Invalid: {invalid}/{len(results_dict)}", invalid/len(results_dict))
        mae = mean_absolute_error(labels, preds)
        rmse = root_mean_squared_error(labels, preds) 
        print(f"MAE: {mae:.4f}; RMSE: {rmse:.4f}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to hidden alignment checkpoint")
    parser.add_argument("--data_path", type=str, default="data/Movies")
    parser.add_argument("--out_dir", type=str, default="results_hidden")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--thinking", type=bool, default=False)

    args = parser.parse_args()
    main(args)
     