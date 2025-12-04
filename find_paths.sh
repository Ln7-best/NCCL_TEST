#!/bin/bash

echo "======================================"
echo "查找 CUDA、NCCL、MPI 库路径"
echo "======================================"

# 1. 查找 CUDA 路径
echo -e "\n[1] 查找 CUDA 路径..."
if command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    echo "✓ 找到 nvcc: $NVCC_PATH"
    echo "✓ CUDA_HOME: $CUDA_HOME"
    nvcc --version | head -n 1
else
    echo "✗ 未找到 nvcc，请检查 CUDA 是否安装"
    echo "  常见安装路径: /usr/local/cuda, /opt/cuda"
fi

# 2. 查找 NCCL 路径
echo -e "\n[2] 查找 NCCL 路径..."
NCCL_PATHS=(
    "/usr/local/cuda-12.6/targets/x86_64-linux/lib"
    "/usr/local/cuda/targets/x86_64-linux/lib"
    "$CUDA_HOME/targets/x86_64-linux/lib"
    "/usr/local/nccl"
    "/usr/lib/x86_64-linux-gnu"
    "$CUDA_HOME/lib64"
    "/opt/nccl"
)

NCCL_FOUND=false
for path in "${NCCL_PATHS[@]}"; do
    if [ -f "$path/libnccl.so" ] || [ -f "$path/lib/libnccl.so" ]; then
        echo "✓ 找到 NCCL 库: $path"
        NCCL_FOUND=true
        # 尝试查找头文件
        if [ -f "$path/include/nccl.h" ]; then
            echo "  头文件: $path/include/nccl.h"
        elif [ -f "$(dirname $path)/include/nccl.h" ]; then
            echo "  头文件: $(dirname $path)/include/nccl.h"
        fi
    fi
done

if [ "$NCCL_FOUND" = false ]; then
    echo "✗ 未找到 NCCL 库"
    echo "  尝试搜索: sudo find / -name 'libnccl.so' 2>/dev/null"
fi

# 3. 查找 MPI 路径
echo -e "\n[3] 查找 MPI 路径..."
if command -v mpirun &> /dev/null; then
    MPIRUN_PATH=$(which mpirun)
    MPI_HOME=$(dirname $(dirname $MPIRUN_PATH))
    echo "✓ 找到 mpirun: $MPIRUN_PATH"
    echo "✓ MPI_HOME: $MPI_HOME"
    mpirun --version | head -n 1
    
    # 查找 MPI 库
    if [ -d "$MPI_HOME/lib" ]; then
        echo "  库路径: $MPI_HOME/lib"
        ls -la "$MPI_HOME/lib/libmpi.so"* 2>/dev/null | head -n 1
    fi
    
    # 查找 MPI 头文件
    if [ -d "$MPI_HOME/include" ]; then
        echo "  头文件路径: $MPI_HOME/include"
    fi
else
    echo "✗ 未找到 mpirun，请检查 MPI 是否安装"
    echo "  注意: /usr/src/kernels 下的 mpi.h 是 Linux 内核的 MPI，不是用户空间的 MPI"
    echo "  安装方法:"
    echo "    Ubuntu/Debian: sudo apt-get install openmpi-bin libopenmpi-dev"
    echo "    CentOS/RHEL: sudo yum install openmpi openmpi-devel"
    echo "    或者使用模块加载: module load openmpi"
fi

# 4. 生成 Makefile 配置建议
echo -e "\n======================================"
echo "Makefile 配置建议:"
echo "======================================"

if [ -n "$CUDA_HOME" ]; then
    echo "CUDA_HOME = $CUDA_HOME"
fi

if [ "$NCCL_FOUND" = true ]; then
    for path in "${NCCL_PATHS[@]}"; do
        if [ -f "$path/libnccl.so" ]; then
            echo "NCCL_HOME = $path"
            break
        elif [ -f "$path/lib/libnccl.so" ]; then
            echo "NCCL_HOME = $(dirname $path)"
            break
        fi
    done
fi

if [ -n "$MPI_HOME" ]; then
    echo "MPI_HOME = $MPI_HOME"
fi

# 5. 检查 GPU 架构
echo -e "\n======================================"
echo "检测 GPU 架构:"
echo "======================================"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU 信息:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | while IFS=',' read -r name cap; do
        name=$(echo $name | xargs)
        cap=$(echo $cap | xargs)
        
        # 转换为 sm_XX 格式
        sm_arch=$(echo $cap | tr -d '.')
        
        echo "  $name"
        echo "    Compute Capability: $cap"
        echo "    CUDA_ARCH 建议: -arch=sm_$sm_arch"
    done
else
    echo "✗ 未找到 nvidia-smi"
fi

echo -e "\n======================================"
echo "完成！"
echo "======================================"
