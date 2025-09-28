import json
import argparse
import logging
import os
from jinja2 import Environment, BaseLoader, FileSystemLoader
from datasets import load_dataset, Dataset, load_from_disk
from typing import Optional, Dict, Union, List
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling 
from trl import SFTTrainer, SFTConfig
import torch
import jsonlines
import numpy as np
from peft import LoraConfig
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
import random

accelerator = Accelerator()



class DataCollatorWithTeacher:
    def __init__(self, base_collator):
        self.base = base_collator
    def __call__(self, features):
        # 去掉，交给基础 collator 处理标准键
        feats = [{k: v for k, v in f.items() if k != "teacher_idx"} for f in features]
        batch = self.base(feats)
        if "teacher_idx" in features[0]:
            t_idx = torch.tensor([f["teacher_idx"] for f in features], dtype=torch.long)
            batch["teacher_idx"] = t_idx
        return batch




def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    import numpy as np

    logits, labels = eval_preds
    preds = np.asarray(logits)
    labels = np.asarray(labels)
    print("Test size:", len(labels))
    # 如果没用 preprocess_logits_for_metrics，这里做一次 argmax 以兼容三维输入
    if preds.ndim == 3:
        preds = preds.argmax(-1)

    # shift for left-to-right causal LM
    preds = preds[..., :-1]
    labels = labels[..., 1:]

    # 你的原始 mask
    mask = (labels != -100) & (labels != 151645)

    # 只在有效 token 上评估
    preds = preds[mask].reshape(-1)
    labels = labels[mask].reshape(-1)
    # np.savez("data_kd/preds_labels.npz", preds=preds, labels=labels)
    # ---- token-level accuracy（原逻辑） ----
    total = labels.size
    accuracy = float((preds == labels).sum() / total) if total > 0 else 0.0

    # ---- MAE / RMSE：仅对答案 token（16..20） ----
    ans_mask = (labels >= 16) & (labels <= 20)
    if ans_mask.any():
        lab_ans = labels[ans_mask].astype(np.int32)
        pred_ans = preds[ans_mask].astype(np.int32)

        # 16..20 -> 1..5
        true_scores = (lab_ans - 15).astype(np.float32)

        in_range = (pred_ans >= 16) & (pred_ans <= 20)
        pred_scores = np.where(in_range, pred_ans - 15, 5).astype(np.float32)

        diff = pred_scores - true_scores
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        n = int(ans_mask.sum())
    else:
        mae = 0.0
        rmse = 0.0
        n = 0

    return {"accuracy": accuracy, "mae": mae, "rmse": rmse, "n": n}

class HiddenDistillSFTTrainer(SFTTrainer):
    def __init__(self,
        teacher_hidden,     # 传入与 train 子集一一对应的 list，每个元素 [L(+1), H]
        num_layers: int,
        hidden_size: int,
        kd_ratio: float = 0.5,
        loss_type: str = "mse",
        # indices: Optional[List[int]] = None,   # ← 不再需要
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.kd_ratio = kd_ratio
        self.loss_type = loss_type

        # 直接保存老师向量（list/tuple/tensor 均可）
        self.teacher_hidden = teacher_hidden
        self.teacher_mask = None

        if hasattr(self.model, "config"):
            self.model.config.output_hidden_states = True

    @staticmethod
    def _last_input_index_from_labels(labels: torch.Tensor) -> torch.Tensor:
        device = labels.device
        first_out = torch.argmax((labels != -100).int(), dim=1)  # [B]
        last_in = (first_out - 1).clamp(min=0)                   # [B]
        return last_in
    
    def _load_teacher_mask_batch(self, batch_size: int, it: int, dp_rank: int, device: torch.device):
        if self.teacher_mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)

        start_idx = dp_rank * batch_size + batch_size * it
        end_idx   = start_idx + batch_size

        mb = self.teacher_mask[start_idx:end_idx]  # 仍在 CPU
        if mb.numel() != batch_size:
            pad = batch_size - mb.numel()
            if pad > 0:
                mb = torch.cat([mb, torch.zeros(pad, dtype=torch.bool)])

        return mb.to(device)
    def _load_teacher_hidden_batch_by_idx(self, idx_B: torch.Tensor, device: torch.device):
        loaded = []
        for i in idx_B.tolist():
            t = self.teacher_hidden[i]
            if not torch.is_tensor(t):
                t = torch.tensor(t)  # 兼容 list/numpy
            loaded.append(t)
        arr = torch.stack(loaded, dim=0)              # [B, L(+1), H] 或 [B, L, H]
        if arr.size(1) == self.num_layers + 1:        # 去掉 embedding 层
            arr = arr[:, 1:, :]
        return arr.to(device, dtype=torch.float32)


    @staticmethod
    def _hidden_distill_loss_masked(student_BLH: torch.Tensor,
                                    teacher_BLH: torch.Tensor,
                                    valid_mask_B: torch.Tensor,
                                    loss_type: str = "mse",
                                    eps: float = 1e-6) -> torch.Tensor:
        # 统一到 float32
        s = student_BLH.to(torch.float32)   # [B, L, H]
        t = teacher_BLH.to(torch.float32)   # [B, L, H]
        B = s.size(0)

        if loss_type == "mse":
            per_layer = F.mse_loss(s, t, reduction='none').mean(dim=-1)  # [B, L]
            per_sample = per_layer.mean(dim=-1)     
        elif loss_type == "mae":
            per_layer = F.l1_loss(s, t, reduction='none').mean(dim=-1)   # [B, L]
            per_sample = per_layer.mean(dim=-1)      
        elif loss_type == "mse_zscore_t":
            mu_t  = t.mean(dim=-1, keepdim=True)                         # [B, L, 1]
            std_t = t.std(dim=-1, keepdim=True).clamp_min(eps)           # [B, L, 1]
            z_s = (s - mu_t) / std_t
            z_t = (t - mu_t) / std_t
            per_vec   = (z_s - z_t).pow(2)                                # [B, L, H]
            per_layer = per_vec.mean(dim=-1)                               # [B, L]
            per_sample = per_layer.mean(dim=-1)                            # [B]
        elif loss_type == "cosine":
            cos_sim = F.cosine_similarity(s, t, dim=-1, eps=eps)  # [B, L]
            per_sample = (1.0 - cos_sim).mean(dim=-1)             # [B]
        elif loss_type == "mse_zscore":
            mu_s  = s.mean(dim=-1, keepdim=True)                         # [B, L, 1]
            std_s = s.std(dim=-1, keepdim=True).clamp_min(eps)
            mu_t  = t.mean(dim=-1, keepdim=True)
            std_t = t.std(dim=-1, keepdim=True).clamp_min(eps)
            z_s = (s - mu_s) / std_s
            z_t = (t - mu_t) / std_t
            per_vec   = (z_s - z_t).pow(2)                                # [B, L, H]
            per_layer = per_vec.mean(dim=-1)                               # [B, L]
            per_sample = per_layer.mean(dim=-1)                            # [B]                
        else:
            std = t.std(dim=-1, keepdim=True).clamp_min(eps)             # [B, L, 1]
            per_vec = (s - t).abs() / std                                 # [B, L, H]
            per_layer = per_vec.mean(dim=-1)                               # [B, L]
            per_sample = per_layer.mean(dim=-1)                            # [B]

        vm = valid_mask_B.to(torch.bool)
        valid_count = vm.sum()
        if valid_count.item() == 0:
            return per_sample.sum() * 0.0

        loss = (per_sample * vm.float()).sum() / valid_count.float()
        return loss  # float32 

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs=False, num_items_in_batch=None):
        mode = "train" if self.model.training else "eval"

        outputs = model(**inputs, output_hidden_states=True)
        lm_loss = outputs.loss

        distil_loss = None
        if self.kd_ratio >= 0 and mode == "train":
            B = inputs["input_ids"].size(0)
            device = (model.module.device if isinstance(model, torch.nn.DataParallel) else model.device)
            teacher_idx = inputs["teacher_idx"]       # [B]
            teacher_hidden_BLH = self._load_teacher_hidden_batch_by_idx(teacher_idx, device)  # [B, L, H]

            valid_mask_B = torch.ones(B, dtype=torch.bool, device=device) 
            hidden_states = outputs.hidden_states  # tuple(len=L+1): [B, T, H]
            assert len(hidden_states) >= self.num_layers + 1

            labels = inputs.get("labels", None)
            if labels is None:
                raise ValueError()
            last_in_idx = self._last_input_index_from_labels(labels)  # [B]

            per_layer = []
            for l in range(1, self.num_layers + 1):  #  1..L
                hs_l = hidden_states[l]  # [B, T, H]
                vec_l = hs_l[torch.arange(hs_l.size(0), device=hs_l.device), last_in_idx, :]  # [B, H]
                per_layer.append(vec_l)
            student_hidden_BLH = torch.stack(per_layer, dim=1)  # [B, L, H]

            distil_loss = self._hidden_distill_loss_masked(
                student_hidden_BLH,
                teacher_hidden_BLH,
                valid_mask_B,
                loss_type=self.loss_type
            ).to(lm_loss.dtype)

            total_loss = (1 - self.kd_ratio) * lm_loss + self.kd_ratio * distil_loss
        else:
            total_loss = lm_loss

        # Logging
        mode_key = "train" if mode == "train" else "eval"
        self._metrics[mode_key]["lm_loss"].append(lm_loss.item())
        if distil_loss is not None:
            self._metrics[mode_key]["distil_loss"].append(distil_loss.item())

        if "labels" in inputs and not self.args.use_liger_kernel:
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()
            mask = shift_labels != -100
            preds = shift_logits.argmax(dim=-1)
            correct = ((preds == shift_labels) & mask).sum()
            total = mask.sum()
            correct = self.accelerator.gather_for_metrics(correct)
            total = self.accelerator.gather_for_metrics(total)
            total_sum = total.sum()
            acc = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode_key]["mean_token_accuracy"].append(acc)

        return (total_loss, outputs) if return_outputs else total_loss


def train(config):
    dataset_name = config["dataset"]["instruction_path"].split("/")[-1]
    model_name = config["models"]["student"].split("/")[-1]
    kd_ratio = config["distillation"]["kd_ratio"]
    lr = config["training"]["learning_rate"]
    epoch = config["training"]["num_train_epochs"]
    ratio = config["dataset"]["training_ratio"]
    loss_type = config["distillation"]["loss_type"]
    mask_name = config["dataset"]["teacher_mask_path"].split("/")[-1].split(".")[0] if config["dataset"]["teacher_mask_path"] else "no_mask"
    exp_name = f"Match-{model_name}_{ratio}-rand-{loss_type}-{mask_name}-hiddenKD@{kd_ratio}-epoch@{epoch}-lr@{str(lr)}-{config['distillation']['peft']}"
    config["training"]["output_dir"] = os.path.join(config["training"]["output_dir"], exp_name)
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    if accelerator.is_main_process:
        wandb.init(project="AmazonReviews-Books-KD", name=exp_name, config=config)
        print("Main process is preparing dataset...")

    dataset = load_from_disk(config["dataset"]["instruction_path"])
    teacher_hidden = torch.load(config["dataset"]["teacher_hidden_path"], map_location="cpu")
    if config["dataset"]["teacher_mask_path"]:
        mask = torch.load(config["dataset"]["teacher_mask_path"])
        mask = np.array(mask, dtype=bool)
    else:
        mask = np.ones(len(dataset["train"]), dtype=bool)
    mask_inds = np.where(mask)[0]
    dataset["train"] = dataset["train"].select(mask_inds)
    teacher_hidden = {str(idx): teacher_hidden[str(ind)]  for idx, ind in enumerate(mask_inds.tolist())}

    student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"], trust_remote_code=True)
    student_tokenizer.padding_side = "left"
    student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], trust_remote_code=True)

    training_arguments = SFTConfig(**config["training"],
                                remove_unused_columns=False,   # ★ 保留 teacher_idx
                                packing=False,                 # ★ 避免样本拼接破坏 1:1 对齐
                                dataloader_drop_last=True,     # 建议，避免最后小 batch 形状不一致
                                ddp_find_unused_parameters=False,  # 你看到的 warning，关掉更省事)
                            )
    peft_module = []
    if "K" in config["distillation"]["peft"]:
        peft_module.append("k_proj")
    if "V" in config["distillation"]["peft"]:
        peft_module.append("v_proj")
    if "Q" in config["distillation"]["peft"]:
        peft_module.append("q_proj")
    if len(peft_module) == 0:
        peft_module = config["distillation"]["peft"]
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=peft_module,  # 只学 KV（和你的思路一致）
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 需要你在 config 里提供 num_layers 和 hidden_size（或从 student_model.config 读）
    num_layers = config["distillation"].get("num_layers", student_model.config.num_hidden_layers)
    hidden_size = config["distillation"].get("hidden_size", student_model.config.hidden_size)

    random.seed(config["dataset"]["seed"])
    num = int(len(dataset["train"]) * ratio)
    indices = random.sample(range(len(dataset["train"])), num)


    # ★ 关键改动：先选子集 -> 构造与之对齐的 teacher_hidden_list -> 写入 teacher_idx
    train_subset = dataset["train"].select(indices)
    teacher_hidden_list = [teacher_hidden[str(i)] for i in indices]   # 与 train_subset 一一对应
    train_subset = train_subset.add_column("teacher_idx", list(range(len(train_subset))))

    pad_token = student_tokenizer.pad_token or student_tokenizer.eos_token
    pad_token_id = student_tokenizer.convert_tokens_to_ids(pad_token)

    use_flash_attention = student_model.config._attn_implementation in [
        "flash_attention_2",
        "flash_attention_3",
        "kernels-community/vllm-flash-attn3",
    ]
    base_collator = DataCollatorForLanguageModeling(pad_token_id=pad_token_id,
                    completion_only_loss=True,)
    data_collator = DataCollatorWithTeacher(base_collator)
    trainer = HiddenDistillSFTTrainer(
        teacher_hidden=teacher_hidden_list,
        num_layers=num_layers,
        hidden_size=hidden_size,
        kd_ratio=config["distillation"]["kd_ratio"],
        data_collator=data_collator, 
        # indices=indices,
        model=student_model,
        processing_class=student_tokenizer,
        args=training_arguments,
        peft_config=peft_config,
        train_dataset=train_subset,
        eval_dataset=dataset["test"],
        loss_type=config["distillation"]["loss_type"],
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    best_path = config["training"]["output_dir"] + "_best"
    trainer.save_model(best_path)
    student_tokenizer.save_pretrained(best_path)


def main():
    p = argparse.ArgumentParser(description="KD Training without config.json")

    # dataset
    p.add_argument("--instruction_path", type=str,
                   default="data/Movies")
    p.add_argument("--teacher_hidden_path", type=str,
                   default="results_train/Movie/last_hidden_dict.pt")
    p.add_argument("--teacher_mask_path", type=str, default="results_train/Movies/mask.pt")

    p.add_argument("--dataset_seed", type=int, default=42)
    p.add_argument("--training_ratio", type=float, default=0.1)

    # distillation
    p.add_argument("--kd_ratio", type=float, default=0.0)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--loss_type", type=str, default="cosine")

    # models
    p.add_argument("--model", type=str,
                   default="ckpt/GRPO_max@10-long@False_step420")

    # training
    p.add_argument("--peft", type=str, default="KV")
    p.add_argument("--output_dir", type=str, default="results_hidden/")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=float, default=200)
    p.add_argument("--eval_steps", type=float, default=200)
    p.add_argument("--eval_strategy", type=str, default="steps")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    p.add_argument("--bf16", default=True)
    p.add_argument("--report_to", type=str, default="wandb")
    p.add_argument("--save_total_limit", type=int, default=30)
    p.add_argument("--metric_for_best_model", type=str, default="eval_accuracy")
    p.add_argument("--greater_is_better", action="store_false", default=True)

    args = p.parse_args()

    config = {
        "dataset": {
            "instruction_path": args.instruction_path,
            "teacher_hidden_path": args.teacher_hidden_path,
            "teacher_mask_path": args.teacher_mask_path,
            "seed": args.dataset_seed,
            "training_ratio": args.training_ratio
            
        },
        "distillation": {
            "kd_ratio": args.kd_ratio,
            "max_seq_length": args.max_seq_length,
            "loss_type": args.loss_type,
            "peft": args.peft
        },
        "models": {
            "teacher": args.model,
            "student": args.model
        },
        "training": {
            "output_dir": args.output_dir,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_length": args.max_length,
            "save_strategy": args.save_strategy,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "eval_strategy": args.eval_strategy,
            "logging_steps": args.logging_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "bf16": args.bf16,
            "report_to": args.report_to,
            "save_total_limit": args.save_total_limit,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
            "load_best_model_at_end": True,
            
        }
    }

    train(config)


if __name__ == "__main__":
    main()