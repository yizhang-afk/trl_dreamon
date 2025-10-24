import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_utils import get_last_checkpoint
import pandas as pd
import numpy as np
from typing import List, Union
import logging
from datetime import datetime
import json

# 导入自定义数据集类
from sft_expand_dataset import SFTExpandDataset

# 设置日志
# --- WandB 上报 ---
# if report_to.lower() == "wandb":
#     try:
#         wandb.init(
#             project="dreamcoder_sft",
#             name=run_name,
#             config={},
#             dir=log_dir
#         )
#         logger.info("WandB 已初始化")
#     except Exception as e:
#         logger.warning(f"WandB 初始化失败: {e}")
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# 创建日志目录
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

# 日志文件名带时间戳
log_file = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 创建 logger
logger = logging.getLogger("dreamcoder_trainer")
logger.setLevel(logging.INFO)

# --- 控制台输出 ---
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# --- 文件输出 ---
file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# 测试
logger.info("日志记录开始")

"""初始化训练器"""
# 自定义数据整理器
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        # 获取批次大小
        batch_size = len(features)
        
        # 提取所有字段
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        position_ids = torch.stack([f["position_ids"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        loss_mask = torch.stack([f["loss_mask"] for f in features])
        t = torch.stack([f["t"] for f in features])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "t": t,
        }
# 自定义训练器类
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        """自定义损失计算，使用loss_mask"""
        labels = inputs.get("labels")
        loss_mask = inputs.get("loss_mask")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = inputs["position_ids"]
        t = inputs["t"]
        # --- 处理 attention_mask ---
        if attention_mask.dim() == 2:
            # (B, S) -> (B, 1, S, S)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1)
            )
        elif attention_mask.dim() == 3:
            # (B, S, S) -> (B, 1, S, S)
            attention_mask = attention_mask.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")
        
        # 前向传播
        # --- 前向传播 ---
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )
        logits = outputs.logits
        shift_logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1).contiguous()
        shift_labels = labels.contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        shift_labels = shift_labels.view(-1)
        loss_mask = loss_mask.reshape(-1)
        loss = loss.masked_fill(~loss_mask, 0)

        # focal 加权
        alpha = 0.25
        gamma = 2
        loss = (
                    alpha
                    * (1 - torch.exp(-loss)) ** gamma
                    * loss
                )
        # time 线性加权
        weight = 1 - t.float().expand(labels.size())
        loss = loss * weight.reshape(-1)
        
        # eos token的降权
        non_eos_mask = (shift_labels != self.tokenizer.eos_token_id) & loss_mask
        non_eos_loss = loss.clone()  
        non_eos_loss[~non_eos_mask] = 0  
        non_eos_count = non_eos_mask.sum().item() 
        non_eos_loss = non_eos_loss.sum()  

        
        eos_mask = (shift_labels == self.tokenizer.eos_token_id) & loss_mask
        eos_loss = loss.clone()  
        eos_loss[~eos_mask] = 0  
        eos_count = eos_mask.sum().item()  
        eos_loss = eos_loss.sum() / eos_count  

        
        loss = (non_eos_loss + eos_loss) / (non_eos_count + 1)

        # num_items_in_batch = inputs["input_ids"].size(0)
        # if num_items_in_batch:
            # loss = loss / num_items_in_batch
        return (loss, outputs) if return_outputs else loss
class DreamCoderTrainer:
    def __init__(
        self,
        model_name_or_path: str,
        train_data_path: str,
        eval_data_path: str,
        output_dir: str = "./dreamcoder_sft_output",
        max_length: int = 1024,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-5,
        num_train_epochs: int = 3,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 100,
        save_total_limit: int = 3,
        dataloader_num_workers: int = 4,
        remove_unused_columns: bool = False,
        report_to: str = "none",
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_loss",
        greater_is_better: bool = False,
        **kwargs
    ):
        self.model_name_or_path = model_name_or_path
        self.train_data_path = train_data_path
        self.eval_data_path = eval_data_path
        self.output_dir = output_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.dataloader_num_workers = dataloader_num_workers
        self.remove_unused_columns = remove_unused_columns
        self.report_to = report_to
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型和tokenizer
        self._setup_model_and_tokenizer()
        
        # 准备数据集
        self._prepare_datasets()
        
        # 设置训练参数
        self._setup_training_args()
        
        # 初始化训练器
        self._setup_trainer()
    
    def _setup_model_and_tokenizer(self):
        """初始化模型和tokenizer"""
        logger.info(f"加载模型和tokenizer: {self.model_name_or_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        
        # 设置特殊token
        # if self.tokenizer.pad_token is None:
            # self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.expand_token_id = 151667
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        # logger.info(f"Using device map: {self.model.device_map}")

        # 设置模型配置
        self.model.config.use_cache = False
        
        logger.info(f"模型参数量: {self.model.num_parameters():,}")
    
    def _prepare_datasets(self):
        """准备训练和验证数据集"""
        logger.info("准备数据集...")
        
        # 训练数据集
        self.train_dataset = SFTExpandDataset(
            parquet_files=[self.train_data_path],
            tokenizer=self.tokenizer,
            prompt_key="prompt",
            response_key="response",
            max_length=self.max_length,
            middle_strategy="line",
            middle_line_num=None,
            merge_prob=0.5,
            merge_schedule="dynamic_inverse",
            use_uniform_merge_prob=0.5
        )
        
        # 验证数据集
        self.eval_dataset = SFTExpandDataset(
            parquet_files=[self.eval_data_path],
            tokenizer=self.tokenizer,
            prompt_key="prompt",
            response_key="response",
            max_length=self.max_length,
            middle_strategy="line",
            middle_line_num=None,
            merge_prob=0.5,
            merge_schedule="dynamic_inverse",
            use_uniform_merge_prob=0.5
        )
        
        logger.info(f"训练样本数: {len(self.train_dataset)}")
        logger.info(f"验证样本数: {len(self.eval_dataset)}")
    
    def _setup_training_args(self):
        """设置训练参数"""
        self.training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            dataloader_num_workers=self.dataloader_num_workers,
            remove_unused_columns=self.remove_unused_columns,
            report_to=self.report_to,
            run_name=f"dreamcoder_sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            seed=42,
            data_seed=42,
            # 全参数微调相关设置
            gradient_checkpointing=True,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            max_grad_norm=1.0,
            # 内存优化
            dataloader_pin_memory=True,
            # 其他设置
            ignore_data_skip=True,
            save_safetensors=True,
        )
    
    def _setup_trainer(self):
        
        
        # 创建数据整理器
        data_collator = CustomDataCollator(self.tokenizer)
        
        # 初始化训练器
        self.trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        # self.trainer = Trainer(
        #     model=self.model,
        #     args=self.training_args,
        #     train_dataset=self.train_dataset,
        #     eval_dataset=self.eval_dataset,
        #     data_collator=data_collator,
        #     tokenizer=self.tokenizer,
        #     compute_loss_func=compute_loss_func, 
        # )
    
    def train(self):
        """开始训练"""
        logger.info("开始训练...")
        
        # 检查是否有检查点
        # last_checkpoint = get_last_checkpoint(self.output_dir)
        # if last_checkpoint:
        #     logger.info(f"从检查点恢复训练: {last_checkpoint}")
        
        # 开始训练
        # train_result = self.trainer.train(resume_from_checkpoint=last_checkpoint if last_checkpoint else None)
        train_result = self.trainer.train()
        # 保存最终模型
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # 保存训练结果
        self.trainer.log_metrics("train", train_result.metrics)
        self.trainer.save_metrics("train", train_result.metrics)
        
        logger.info("训练完成!")
        return train_result
    
    def evaluate(self):
        """评估模型"""
        logger.info("开始评估...")
        
        eval_result = self.trainer.evaluate()
        
        # 保存评估结果
        self.trainer.log_metrics("eval", eval_result)
        self.trainer.save_metrics("eval", eval_result)
        
        logger.info(f"评估结果: {eval_result}")
        return eval_result
    
    def save_training_config(self):
        """保存训练配置"""
        config = {
            "model_name_or_path": self.model_name_or_path,
            "train_data_path": self.train_data_path,
            "eval_data_path": self.eval_data_path,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
        }
        
        config_path = os.path.join(self.output_dir, "training_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练配置已保存到: {config_path}")


def main():
    """主函数"""
    # 配置参数
    model_name_or_path = "Dream-org/Dream-Coder-v0-Base-7B"
    train_data_path = "data/opencoder-stage2-edu/train_data.parquet"
    eval_data_path = "sdata/opencoder-stage2-edu/eval_data.parquet"
    output_dir = "./dreamcoder_sft_output"
    
    # 创建训练器
    trainer = DreamCoderTrainer(
        model_name_or_path=model_name_or_path,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        output_dir=output_dir,
        max_length=1024,
        batch_size=32,  # 根据GPU显存调整
        gradient_accumulation_steps=8,  # 有效批次大小 = 2 * 8 = 16
        learning_rate=1e-5,
        num_train_epochs=3,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_steps=20,
        eval_steps=10,
        logging_steps=1,
        save_total_limit=3,
        dataloader_num_workers=4,
    )
    
    # 保存训练配置
    trainer.save_training_config()
    
    # 开始训练
    train_result = trainer.train()
    
    # 评估模型
    eval_result = trainer.evaluate()
    
    logger.info("训练和评估完成!")
    logger.info(f"训练结果: {train_result.metrics}")
    logger.info(f"评估结果: {eval_result}")


if __name__ == "__main__":
    main()
