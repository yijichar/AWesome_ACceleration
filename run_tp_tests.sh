#!/bin/bash
# 快速测试脚本

echo "=========================================="
echo "Tensor Parallel 测试脚本"
echo "=========================================="

# 检查 CUDA 可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到 NVIDIA GPU"
    exit 1
fi

# 获取 GPU 数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "检测到 $GPU_COUNT 个 GPU"

# 测试 1: 单卡 baseline
echo ""
echo "=========================================="
echo "测试 1: 单卡 baseline"
echo "=========================================="
python test_tp.py

# 测试 2: 2 卡 TP
if [ $GPU_COUNT -ge 2 ]; then
    echo ""
    echo "=========================================="
    echo "测试 2: 2 卡 Tensor Parallel"
    echo "=========================================="
    torchrun --nproc_per_node=2 test_tp.py
fi

# 测试 3: 4 卡 TP
if [ $GPU_COUNT -ge 4 ]; then
    echo ""
    echo "=========================================="
    echo "测试 3: 4 卡 Tensor Parallel"
    echo "=========================================="
    torchrun --nproc_per_node=4 test_tp.py
fi

echo ""
echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
