#!/usr/bin/env python3
"""
使用 mpi4py 启动 NCCL 测试程序（无 MPI 版本）
安装: pip install mpi4py
运行: mpiexec -n 8 python3 run_nccl_test.py
"""

from mpi4py import MPI
import subprocess
import sys
import os

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"使用 mpi4py 启动 NCCL 测试（无 MPI 版本）")
        print(f"进程数: {size}")
        print("=" * 50)
    
    # 设置环境变量，让 C++ 程序知道自己的 rank 和 size
    env = os.environ.copy()
    env['OMPI_COMM_WORLD_RANK'] = str(rank)
    env['OMPI_COMM_WORLD_SIZE'] = str(size)
    
    # 启动 C++ 程序（无 MPI 版本）
    try:
        result = subprocess.run(
            ['./nccl_split_test'],
            env=env,
            check=True,
            capture_output=False
        )
    except subprocess.CalledProcessError as e:
        print(f"Rank {rank}: 程序执行失败，错误码: {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        if rank == 0:
            print("错误: 找不到 nccl_split_test 可执行文件", file=sys.stderr)
            print("请先运行: make", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
