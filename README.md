# Token-Efficient Long-Term Interest Sketching and Internalized Reasoning for LLM-based Recommendation

This repository contains the official codebase for our paper:  
**"Token-Efficient Long-Term Interest Sketching and Internalized Reasoning for LLM-based Recommendation."**

---

## 🚀 Requirements

To install the required dependencies:

```bash
mkdir logs
mkdir results
mkdir ckpt
pip3 install -e .
pip install -r requirements.txt
```

##  📊 Datasets
The following table summarizes the datasets used in our experiments:

| Dataset | # Train | # Valid | # Test | # User | # Item |
|---------|---------|---------|--------|--------|--------|
| Book    | 94075   | 12222   | 11708  | 10440  | 9753   |
| Movie   | 22097   | 2629    | 2629   | 2629   | 10874  |


We provide the raw JSONL for the **Movies** dataset in `./data/`. The **Books** dataset is available at [this repository](https://github.com/jieyong99/EXP3RT/tree/main/data/amazon-book/rating_bias).


## 🚀 Quickstart
###  1. Generate Long-term Interest Sketch
```bash
python generate_interest_sketch.py \
  --data_category Movies \
  --meta_path data/Movies/meta_Movies_and_TV.jsonl.gz \
  --history_path data/Movies/history.jsonl.gz \
  --output_dir data/Movies \
  --n_clusters 20 \
  --top_k 100
```
> **Note:** 
  * The needed meta data in Amazon Reviews datasets can be downloaded from [Books](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz), and [Movies](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Movies_and_TV.jsonl.gz).
  * We provide the preprocessed data in `./data/`.


###  2. Preparation for GRPO
Before GRPO training, preprocess the datasets using the following command:
```bash
  python preprocess_grpo.py \
    --category Movies \
    --data_root /mnt/hdfs/zhihao/AmazonReviews \
    --max_history 10 \
    --use_long_history False 
```

### 3. 🔥 GRPO Training
```bash
  bash main_grpo_amazon.sh
```

### 4. Preparation for Hidden Alignment
#### (a) Generate GRPO reasoning on the train set
```bash
  accelerate launch --multi_gpu inference_train_ddp.py \
    --model_path ckpt/GRPO_max@10-long@False_step420 \
    --data_path data/Movies/verl/max@10-long@False 
```
#### (b) Extract hidden states at the <answer> token (when both prompt and CoT are in the input):
```bash
  accelerate launch --multi_gpu extract_last_hidden_ddp.py \
    --model ckpt/GRPO_max@10-long@False_step420 \
    --thinking_path results_train/Movies\
    --data_path data/Movies/verl \
    --batch_size 64 \
    --out_dir results_train/Movies \
```
#### (c) Preprocess dataset for hidden alignment:
```bash
  python preprocess_hidden.py \
    --category Movies \
    --max_history 10 
```
### 5. 🔥 Hidden Alignment
```bash
  accelerate launch --config_file config_Zero2.yaml train_hidden_mask.py \
      --instruction_path data/Movies \
      --teacher_hidden_path results_train/Movie/last_hidden_dict.pt \
      --teacher_mask_path results_train/Movies/mask.pt \
      --model ckpt/GRPO_max@10-long@False_step420 \
      --output_dir results_hidden/
```
### 6. 🔎 Inference on the test set (Answer-Only)
```bash
  accelerate launch --multi_gpu inference_ao_ddp.py \
      --model [PATH_TO_HIDDEN_ALIGNMENT_CHECKPOINT] \
      --data_path data/Movies \
```


## 📦 Repository structure (key files)
.
├── data/
├── scripts/                # optional shell helpers
├── generate_interest_sketch.py
├── preprocess_grpo.py
├── preprocess_hidden.py
├── inference_train_ddp.py
├── extract_last_hidden_ddp.py
├── train_hidden_mask.py
├── inference_ao_ddp.py
├── requirements.txt
└── setup.py



## 🙏 Acknowledgement
We build upon excellent open-source tools:
* [VERL](https://github.com/megagonlabs/verl)
* [TRL](https://github.com/huggingface/trl)
* [EasyDistill](https://github.com/modelscope/easydistill/tree/main)
