#!/usr/bin/env python3
"""
简单的多进程启动脚本（不依赖 mpiexec）
直接启动 8 个进程运行 NCCL 测试
"""

import subprocess
import sys
import os
import time

def main():
    num_processes = 8
    processes = []
    
    print(f"启动 {num_processes} 个进程...")
    print("=" * 50)
    
    # 启动 8 个进程
    for rank in range(num_processes):
        env = os.environ.copy()
        env['OMPI_COMM_WORLD_RANK'] = str(rank)
        env['OMPI_COMM_WORLD_SIZE'] = str(num_processes)
        
        # 启动进程
        proc = subprocess.Popen(
            ['./nccl_split_test'],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(proc)
        print(f"✓ 进程 {rank} 已启动 (PID: {proc.pid})")
    
    print("\n等待所有进程完成...")
    print("=" * 50)
    
    # 等待所有进程完成
    all_success = True
    for rank, proc in enumerate(processes):
        stdout, stderr = proc.communicate()
        
        if proc.returncode == 0:
            print(f"\n[进程 {rank}] 输出:")
            if stdout:
                print(stdout)
        else:
            print(f"\n[进程 {rank}] 失败 (返回码: {proc.returncode})")
            all_success = False
            if stderr:
                print(f"错误信息:\n{stderr}")
    
    print("\n" + "=" * 50)
    if all_success:
        print("✓ 所有进程执行成功！")
        return 0
    else:
        print("✗ 部分进程执行失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())
