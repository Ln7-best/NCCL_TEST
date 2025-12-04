#include <nccl.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

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

// 获取所有进程中的最大时间
double get_max_time(double local_time, int rank, int size) {
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return max_time;
}

// 测试直接创建7卡通信组
void test_direct_7gpu_comm(int rank, int size) {
    if (rank == 0) {
        std::cout << "\n=== Test 1: 直接创建 7 卡通信组 ===" << std::endl;
    }
    
    // 只有前7个rank参与
    if (rank >= 7) {
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
    
    // 设置当前进程使用的GPU
    CUDACHECK(cudaSetDevice(rank));
    
    ncclComm_t comm;
    ncclUniqueId id;
    cudaStream_t stream;
    
    Timer timer;
    
    // Rank 0 生成唯一ID并广播
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // 所有7个rank同步后开始计时
    MPI_Barrier(MPI_COMM_WORLD);
    timer.tic();
    
    // 直接创建7卡通信组
    NCCLCHECK(ncclCommInitRank(&comm, 7, id, rank));
    
    double local_init_time = timer.toc();
    double init_time = get_max_time(local_init_time, rank, size);
    
    if (rank == 0) {
        std::cout << "通信组创建时间: " << std::fixed << std::setprecision(3) 
                  << init_time << " ms" << std::endl;
    }
    
    // 创建stream并准备数据
    CUDACHECK(cudaStreamCreate(&stream));
    size_t dataSize = 64 * 1024 * 1024; // 64MB
    float* d_data;
    CUDACHECK(cudaMalloc(&d_data, dataSize * sizeof(float)));
    CUDACHECK(cudaMemset(d_data, 1, dataSize * sizeof(float)));
    
    // 同步后测试第一次AllReduce
    MPI_Barrier(MPI_COMM_WORLD);
    timer.tic();
    
    NCCLCHECK(ncclAllReduce(d_data, d_data, dataSize, 
                            ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    
    double local_allreduce_time = timer.toc();
    double allreduce_time = get_max_time(local_allreduce_time, rank, size);
    
    if (rank == 0) {
        double data_gb = (dataSize * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
        double bus_bandwidth = data_gb * 2 * 6 / 7 / (allreduce_time / 1000.0);
        
        std::cout << "第一次 AllReduce 时间: " << allreduce_time << " ms" << std::endl;
        std::cout << "数据大小: " << data_gb << " GB" << std::endl;
        std::cout << "算法带宽 (Algbw): " << bus_bandwidth << " GB/s" << std::endl;
    }
    
    // 清理
    CUDACHECK(cudaFree(d_data));
    CUDACHECK(cudaStreamDestroy(stream));
    ncclCommDestroy(comm);
    
    MPI_Barrier(MPI_COMM_WORLD);
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
    
    // Rank 0 生成唯一ID并广播
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // 所有8个rank同步后开始计时
    MPI_Barrier(MPI_COMM_WORLD);
    timer.tic();
    
    // 先创建8卡通信组
    NCCLCHECK(ncclCommInitRank(&comm_8, 8, id, rank));
    
    double local_init_8gpu_time = timer.toc();
    double init_8gpu_time = get_max_time(local_init_8gpu_time, rank, size);
    
    if (rank == 0) {
        std::cout << "8卡通信组创建时间: " << std::fixed << std::setprecision(3) 
                  << init_8gpu_time << " ms" << std::endl;
    }
    
    // 从8卡split出7卡（排除第7号卡,即rank 7）
    MPI_Barrier(MPI_COMM_WORLD);
    timer.tic();
    
    int color = (rank == 7) ? NCCL_SPLIT_NOCOLOR : 0; // rank 7不参与
    NCCLCHECK(ncclCommSplit(comm_8, color, rank, &comm_7, NULL));
    
    double local_split_time = timer.toc();
    double split_time = get_max_time(local_split_time, rank, size);
    
    if (rank == 0) {
        std::cout << "CommSplit 时间: " << split_time << " ms" << std::endl;
        std::cout << "总创建时间 (8卡+split): " << (init_8gpu_time + split_time) << " ms" << std::endl;
    }
    
    // 只有前7个rank在7卡通信组上测试AllReduce
    if (rank < 7) {
        CUDACHECK(cudaStreamCreate(&stream));
        size_t dataSize = 64 * 1024 * 1024; // 64MB
        float* d_data;
        CUDACHECK(cudaMalloc(&d_data, dataSize * sizeof(float)));
        CUDACHECK(cudaMemset(d_data, 1, dataSize * sizeof(float)));
        
        // 同步后测试第一次AllReduce
        MPI_Barrier(MPI_COMM_WORLD);
        timer.tic();
        
        NCCLCHECK(ncclAllReduce(d_data, d_data, dataSize, 
                                ncclFloat, ncclSum, comm_7, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        
        double local_allreduce_time = timer.toc();
        double allreduce_time = get_max_time(local_allreduce_time, rank, size);
        
        if (rank == 0) {
            double data_gb = (dataSize * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
            double bus_bandwidth = data_gb * 2 * 6 / 7 / (allreduce_time / 1000.0);
            
            std::cout << "第一次 AllReduce 时间: " << allreduce_time << " ms" << std::endl;
            std::cout << "数据大小: " << data_gb << " GB" << std::endl;
            std::cout << "算法带宽 (Algbw): " << bus_bandwidth << " GB/s" << std::endl;
        }
        
        // 清理
        CUDACHECK(cudaFree(d_data));
        CUDACHECK(cudaStreamDestroy(stream));
        ncclCommDestroy(comm_7);
    } else {
        // rank 7 等待其他rank完成
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    ncclCommDestroy(comm_8);
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        std::cout << "NCCL CommSplit 性能测试 (多进程版本)" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "MPI 进程数: " << size << std::endl;
    }
    
    // 检查进程数
    if (size != 8) {
        if (rank == 0) {
            std::cerr << "错误: 需要恰好8个MPI进程！" << std::endl;
            std::cerr << "请使用: mpirun -np 8 ./nccl_split_test" << std::endl;
        }
        MPI_Finalize();
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
        MPI_Finalize();
        return 1;
    }
    
    // 测试1: 直接创建7卡通信组
    test_direct_7gpu_comm(rank, size);
    
    // 测试2: 从8卡split出7卡
    test_split_7gpu_from_8gpu(rank, size);
    
    if (rank == 0) {
        std::cout << "\n=== 总结 ===" << std::endl;
        std::cout << "请对比以上两种方式的通信组创建时间和第一次AllReduce性能" << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
