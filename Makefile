# NCCL Split Test Makefile (完全无 MPI 版本)

# 编译器设置
NVCC = nvcc

# 编译选项
CUDA_ARCH = -arch=sm_90  # 根据你的GPU架构调整
NVCC_FLAGS = -O3 -std=c++11 $(CUDA_ARCH)

# 库路径
NCCL_HOME ?= /usr/local/cuda-12.6/targets/x86_64-linux
CUDA_HOME ?= /usr/local/cuda

# 包含路径（不需要 MPI）
INCLUDES = -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include

# 库链接（不需要 MPI）
LIBS = -L$(NCCL_HOME)/lib -L$(CUDA_HOME)/lib64 -lnccl -lcudart -Xlinker -rpath -Xlinker $(NCCL_HOME)/lib

# 目标文件
TARGET = nccl_split_test

all: $(TARGET)

$(TARGET): nccl_split_test.cu
	@echo "编译无 MPI 版本..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@ $(LIBS)
	@echo "✓ 编译成功"

clean:
	rm -f $(TARGET)

.PHONY: all clean
