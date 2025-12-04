#!/bin/bash

# NCCL Split Test 运行脚本

echo "======================================"
echo "NCCL CommSplit 性能测试"
echo "======================================"

# 检查是否已编译
if [ ! -f "./nccl_split_test" ]; then
    echo "程序未编译，正在编译..."
    make
    if [ $? -ne 0 ]; then
        echo "编译失败！"
        exit 1
    fi
fi

# 设置环境变量（根据需要调整）
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# 运行测试（8个进程对应8张卡）
echo ""
echo "启动测试（8个MPI进程）..."
mpirun -np 8 \
    --allow-run-as-root \
    -x NCCL_DEBUG \
    -x NCCL_IB_DISABLE \
    -x NCCL_NET_GDR_LEVEL \
    ./nccl_split_test

echo ""
echo "测试完成！"
