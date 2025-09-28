from transformers import PreTrainedModel, PreTrainedTokenizerBase,AutoModelForCausalLM, AutoTokenizer, TrainingArguments
tokenizer = AutoTokenizer.from_pretrained("Qwen3-4B", padding_side="left")
tokenizer.add_tokens(["<answer>", "</answer>"])
tokenizer.save_pretrained("Qwen3-4B")
model = AutoModelForCausalLM.from_pretrained("Qwen3-4B")
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained("Qwen3-4B")