#!/bin/bash

# DreamCoder7B 全参数微调训练脚本
# 使用方法: bash run_training.sh

echo "开始DreamCoder7B全参数微调训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一块GPU，如果有多个GPU可以调整
export TOKENIZERS_PARALLELISM=false  # 避免tokenizer警告

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查CUDA是否可用
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU数量: {torch.cuda.device_count()}')"

# 安装依赖（如果需要）
echo "检查并安装依赖..."
pip install -r requirements.txt

# 创建输出目录
mkdir -p ./dreamcoder_sft_output
mkdir -p ./dreamcoder_sft_output/logs

# 开始训练
echo "开始训练..."
python sft_train.py

echo "训练完成！"
echo "模型保存在: ./dreamcoder_sft_output/"
echo "日志保存在: ./dreamcoder_sft_output/logs/"
