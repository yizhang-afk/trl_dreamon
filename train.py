import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# -----------------------------
# 配置
# -----------------------------
MODEL_NAME = "/Users/lris/Desktop/HIT/鲸鱼科技/dreamOn/sft_training/model/Dream-Coder-v0-Base-7B"
DATA_PATH = "/Users/lris/Desktop/HIT/鲸鱼科技/trl/data/opencoder-stage2-edu"
OUTPUT_DIR = "./dreamcoder7b-finetuned"
MAX_LENGTH = 512
BATCH_SIZE = 1             # 根据显存调节
GRAD_ACCUM_STEPS = 8
EPOCHS = 3
LR = 2e-5
BF16 = True                 # 如果 GPU 支持 bf16

# -----------------------------
# 数据加载
# -----------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": os.path.join(DATA_PATH, "train.jsonl"),
        "validation": os.path.join(DATA_PATH, "val.jsonl")
    }
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(batch):
    inputs = tokenizer(batch["input"], truncation=True, max_length=MAX_LENGTH)
    labels = tokenizer(batch["output"], truncation=True, max_length=MAX_LENGTH)
    batch["input_ids"] = inputs["input_ids"]
    batch["attention_mask"] = inputs["attention_mask"]
    batch["labels"] = labels["input_ids"]
    return batch

tokenized_dataset = dataset.map(tokenize_fn, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------------
# 模型加载
# -----------------------------
device_map = "auto" if torch.cuda.is_available() else None
torch_dtype = torch.bfloat16 if BF16 else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device_map,
    torch_dtype=torch_dtype
)

# -----------------------------
# 训练配置
# -----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    bf16=BF16,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    load_best_model_at_end=True,
    fp16=False,
)

# -----------------------------
# TRL SFTTrainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    args=training_args
)

# -----------------------------
# 开始训练
# -----------------------------
trainer.train()

# -----------------------------
# 保存模型
# -----------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("模型训练完成并已保存到:", OUTPUT_DIR)