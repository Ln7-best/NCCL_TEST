#!/bin/bash

echo "======================================"
echo "NCCL 测试环境设置和运行脚本（无 MPI 版本）"
echo "======================================"

# 1. 检查 Python
echo -e "\n[1] 检查 Python 环境..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ 找到 Python: $PYTHON_VERSION"
else
    echo "✗ 未找到 Python3"
    exit 1
fi

# 2. 安装 mpi4py
echo -e "\n[2] 安装 mpi4py..."
pip3 install mpi4py --user
if [ $? -eq 0 ]; then
    echo "✓ mpi4py 安装成功"
else
    echo "✗ mpi4py 安装失败"
    exit 1
fi

# 3. 编译 NCCL 测试程序（无 MPI 版本）
echo -e "\n[3] 编译 NCCL 测试程序（无 MPI 版本）..."
make clean
make
if [ $? -eq 0 ]; then
    echo "✓ 编译成功"
else
    echo "✗ 编译失败"
    exit 1
fi

# 4. 运行测试
echo -e "\n[4] 运行测试..."
echo "======================================"
# 尝试使用 mpiexec，如果不存在则使用简单的 Python 启动脚本
if command -v mpiexec &> /dev/null; then
    mpiexec -n 8 python3 run_nccl_test.py
elif command -v mpirun &> /dev/null; then
    mpirun -np 8 python3 run_nccl_test.py
else
    echo "未找到 mpiexec/mpirun，使用简单启动脚本..."
    python3 run_test_simple.py
fi

echo -e "\n======================================"
echo "测试完成！"
echo "======================================"
