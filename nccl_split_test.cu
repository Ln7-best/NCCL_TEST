#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <unistd.h>

// 从环境变量获取 rank 和 size
int get_rank() {
    const char* rank_str = std::getenv("OMPI_COMM_WORLD_RANK");
    if (!rank_str) rank_str = std::getenv("PMI_RANK");
    if (!rank_str) rank_str = std::getenv("SLURM_PROCID");
    return rank_str ? std::atoi(rank_str) : 0;
}

int get_world_size() {
    const char* size_str = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!size_str) size_str = std::getenv("PMI_SIZE");
    if (!size_str) size_str = std::getenv("SLURM_NTASKS");
    return size_str ? std::atoi(size_str) : 1;
}

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                           \
  if (err != cudaSuccess) {                        \
    std::cerr << "Rank " << rank << " CUDA error " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE);                            \
  }                                                \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                          \
  if (res != ncclSuccess) {                        \
    std::cerr << "Rank " << rank << " NCCL error " << ncclGetErrorString(res) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE);                            \
  }                                                \
} while(0)

class Timer {
    std::chrono::high_resolution_clock::time_point start;
public:
    void tic() { start = std::chrono::high_resolution_clock::now(); }
    double toc() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
};

// 简单的进程间同步：通过文件系统
void barrier(int rank, int size, const std::string& barrier_name) {
    std::string barrier_dir = "/tmp/nccl_barrier_" + barrier_name;
    std::string my_file = barrier_dir + "/rank_" + std::to_string(rank);
    
    // Rank 0 创建目录
    if (rank == 0) {
        system(("mkdir -p " + barrier_dir).c_str());
    }
    usleep(100000); // 等待目录创建
    
    // 每个进程创建自己的文件
    std::ofstream(my_file).close();
    
    // 等待所有进程到达
    while (true) {
        int count = 0;
        for (int i = 0; i < size; i++) {
            std::string file = barrier_dir + "/rank_" + std::to_string(i);
            std::ifstream f(file);
            if (f.good()) count++;
        }
        if (count == size) break;
        usleep(10000); // 10ms
    }
    
    // Rank 0 清理
    if (rank == 0) {
        system(("rm -rf " + barrier_dir).c_str());
    }
}

// 广播 NCCL unique ID
void broadcast_nccl_id(ncclUniqueId* id, int rank, int size) {
    std::string id_file = "/tmp/nccl_unique_id";
    
    if (rank == 0) {
        // Rank 0 生成并写入文件
        ncclGetUniqueId(id);
        std::ofstream f(id_file, std::ios::binary);
        f.write(reinterpret_cast<char*>(id), sizeof(ncclUniqueId));
        f.close();
    }
    
    barrier(rank, size, "id_broadcast");
    
    if (rank != 0) {
        // 其他 rank 读取
        std::ifstream f(id_file, std::ios::binary);
        f.read(reinterpret_cast<char*>(id), sizeof(ncclUniqueId));
        f.close();
    }
    
    barrier(rank, size, "id_read");
    
    if (rank == 0) {
        unlink(id_file.c_str());
    }
}

// 获取所有进程中的最大时间
double get_max_time(double local_time, int rank, int size) {
    // 写入本地时间
    std::string time_file = "/tmp/nccl_time_rank_" + std::to_string(rank);
    std::ofstream f(time_file);
    f << std::fixed << std::setprecision(6) << local_time;
    f.close();
    
    barrier(rank, size, "time_reduce");
    
    // Rank 0 收集所有时间
    double max_time = local_time;
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            std::string file = "/tmp/nccl_time_rank_" + std::to_string(i);
            std::ifstream inf(file);
            double t;
            inf >> t;
            if (t > max_time) max_time = t;
            inf.close();
            unlink(file.c_str());
        }
        // 写入最大时间
        std::ofstream outf("/tmp/nccl_max_time");
        outf << std::fixed << std::setprecision(6) << max_time;
        outf.close();
    }
    
    barrier(rank, size, "time_broadcast");
    
    // 所有进程读取最大时间
    if (rank != 0) {
        std::ifstream inf("/tmp/nccl_max_time");
        inf >> max_time;
        inf.close();
    }
    
    barrier(rank, size, "time_read");
    
    if (rank == 0) {
        unlink("/tmp/nccl_max_time");
    }
    
    return max_time;
}

// 测试直接创建7卡通信组
void test_direct_7gpu_comm(int rank, int size) {
    if (rank == 0) {
        std::cout << "\n=== Test 1: 直接创建 7 卡通信组 ===" << std::endl;
    }
    
    ncclUniqueId id;
    
    // 广播 unique ID - 所有进程都要参与
    broadcast_nccl_id(&id, rank, size);
    
    // 只有前7个rank参与后续操作
    if (rank >= 7) {
        barrier(rank, size, "test1_start");
        barrier(rank, size, "test1_allreduce");
        barrier(rank, size, "test1_end");
        return;
    }
    
    // 设置当前进程使用的GPU
    CUDACHECK(cudaSetDevice(rank));
    
    ncclComm_t comm;
    cudaStream_t stream;
    
    Timer timer;
    
    // 同步后开始计时
    barrier(rank, size, "test1_start");
    timer.tic();
    
    // 直接创建7卡通信组
    NCCLCHECK(ncclCommInitRank(&comm, 7, id, rank));
    
    double local_init_time = timer.toc();
    
    // 创建stream并准备数据
    CUDACHECK(cudaStreamCreate(&stream));
    size_t dataSize = 64 * 1024 * 1024; // 64MB
    float* d_data;
    CUDACHECK(cudaMalloc(&d_data, dataSize * sizeof(float)));
    CUDACHECK(cudaMemset(d_data, 1, dataSize * sizeof(float)));
    
    // 同步后测试第一次AllReduce
    barrier(rank, size, "test1_allreduce");
    timer.tic();
    
    NCCLCHECK(ncclAllReduce(d_data, d_data, dataSize, 
                            ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    
    double local_allreduce_time = timer.toc();
    
    if (rank == 0) {
        double data_gb = (dataSize * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
        
        std::cout << "通信组创建时间: " << std::fixed << std::setprecision(3) 
                  << local_init_time << " ms" << std::endl;
        std::cout << "第一次 AllReduce 时间: " << local_allreduce_time << " ms" << std::endl;
        std::cout << "数据大小: " << data_gb << " GB" << std::endl;
        
        double bus_bandwidth = data_gb * 2 * 6 / 7 / (local_allreduce_time / 1000.0);
        std::cout << "算法带宽 (Algbw): " << bus_bandwidth << " GB/s" << std::endl;
    }
    
    // 清理
    CUDACHECK(cudaFree(d_data));
    CUDACHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    
    barrier(rank, size, "test1_end");
}

// 测试从8卡通信组split出7卡通信组
void test_split_7gpu_from_8gpu(int rank, int size) {
    if (rank == 0) {
        std::cout << "\n=== Test 2: 从8卡通信组split出7卡通信组 ===" << std::endl;
    }
    
    // 设置当前进程使用的GPU
    CUDACHECK(cudaSetDevice(rank));
    
    ncclComm_t comm_8, comm_7;
    ncclUniqueId id;
    cudaStream_t stream;
    
    Timer timer;
    
    // 广播 unique ID
    broadcast_nccl_id(&id, rank, size);
    
    // 同步后开始计时
    barrier(rank, size, "test2_start");
    timer.tic();
    
    // 先创建8卡通信组
    NCCLCHECK(ncclCommInitRank(&comm_8, 8, id, rank));
    
    double local_init_8gpu_time = timer.toc();
    
    if (rank == 0) {
        std::cout << "8卡通信组创建时间: " << std::fixed << std::setprecision(3) 
                  << local_init_8gpu_time << " ms" << std::endl;
    }
    
    // 从8卡split出7卡（排除第7号卡,即rank 7）
    barrier(rank, size, "test2_split");
    timer.tic();
    
    int color = (rank == 7) ? NCCL_SPLIT_NOCOLOR : 0; // rank 7不参与
    NCCLCHECK(ncclCommSplit(comm_8, color, rank, &comm_7, NULL));
    
    double local_split_time = timer.toc();
    
    if (rank == 0) {
        std::cout << "CommSplit 时间: " << local_split_time << " ms" << std::endl;
        std::cout << "总创建时间 (8卡+split): " << (local_init_8gpu_time + local_split_time) << " ms" << std::endl;
    }
    
    // 只有前7个rank在7卡通信组上测试AllReduce
    if (rank < 7) {
        CUDACHECK(cudaStreamCreate(&stream));
        size_t dataSize = 64 * 1024 * 1024; // 64MB
        float* d_data;
        CUDACHECK(cudaMalloc(&d_data, dataSize * sizeof(float)));
        CUDACHECK(cudaMemset(d_data, 1, dataSize * sizeof(float)));
        
        // 同步后测试第一次AllReduce
        barrier(rank, size, "test2_allreduce");
        timer.tic();
        
        NCCLCHECK(ncclAllReduce(d_data, d_data, dataSize, 
                                ncclFloat, ncclSum, comm_7, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        
        double local_allreduce_time = timer.toc();
        
        if (rank == 0) {
            double data_gb = (dataSize * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
            double bus_bandwidth = data_gb * 2 * 6 / 7 / (local_allreduce_time / 1000.0);
            
            std::cout << "第一次 AllReduce 时间: " << local_allreduce_time << " ms" << std::endl;
            std::cout << "数据大小: " << data_gb << " GB" << std::endl;
            std::cout << "算法带宽 (Algbw): " << bus_bandwidth << " GB/s" << std::endl;
        }
        
        // 清理
        CUDACHECK(cudaFree(d_data));
        CUDACHECK(cudaStreamDestroy(stream));
        ncclCommDestroy(comm_7);
    } else {
        // rank 7 等待其他rank完成
        barrier(rank, size, "test2_allreduce");
    }
    
    ncclCommDestroy(comm_8);
    barrier(rank, size, "test2_end");
}

int main(int argc, char* argv[]) {
    int rank = get_rank();
    int size = get_world_size();
    
    if (rank == 0) {
        std::cout << "NCCL CommSplit 性能测试 (无 MPI 版本)" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "进程数: " << size << std::endl;
    }
    
    // 检查进程数
    if (size != 8) {
        if (rank == 0) {
            std::cerr << "错误: 需要恰好8个进程！" << std::endl;
        }
        return 1;
    }
    
    // 检查GPU数量
    int nGPUs;
    CUDACHECK(cudaGetDeviceCount(&nGPUs));
    if (rank == 0) {
        std::cout << "检测到 " << nGPUs << " 张GPU" << std::endl;
    }
    
    if (nGPUs < 8) {
        if (rank == 0) {
            std::cerr << "需要至少8张GPU才能运行此测试！" << std::endl;
        }
        return 1;
    }
    
    // 运行测试
    test_direct_7gpu_comm(rank, size);
    test_split_7gpu_from_8gpu(rank, size);
    
    if (rank == 0) {
        std::cout << "\n测试完成！" << std::endl;
    }
    
    return 0;
}
