# NCCL Split Test Makefile

# 编译器设置
NVCC = nvcc

# 编译选项
CUDA_ARCH = -arch=sm_90  # 根据你的GPU架构调整，如 sm_70, sm_75, sm_80, sm_86, sm_89, sm_90
NVCC_FLAGS = -O3 -std=c++11 $(CUDA_ARCH)

# 库路径（根据实际情况调整）
NCCL_HOME ?= /usr/local/nccl
CUDA_HOME ?= /usr/local/cuda
MPI_HOME ?= /usr/local/openmpi

# 包含路径
INCLUDES = -I$(NCCL_HOME)/include -I$(CUDA_HOME)/include -I$(MPI_HOME)/include

# 库链接
LIBS = -L$(NCCL_HOME)/lib -L$(CUDA_HOME)/lib64 -L$(MPI_HOME)/lib -lnccl -lcudart -lmpi

# 目标文件
TARGET = nccl_split_test

all: $(TARGET)

$(TARGET): nccl_split_test.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@ $(LIBS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	mpirun -np 8 ./$(TARGET)

.PHONY: all clean run
