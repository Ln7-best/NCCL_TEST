#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                           \
  if (err != cudaSuccess) {                        \
    std::cerr << "CUDA error " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE);                            \
  }                                                \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                          \
  if (res != ncclSuccess) {                        \
    std::cerr << "NCCL error " << ncclGetErrorString(res) \
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

// 测试直接创建7卡通信组
void test_direct_7gpu_comm(int nGPUs = 7) {
    std::cout << "\n=== Test 1: 直接创建 " << nGPUs << " 卡通信组 ===" << std::endl;
    
    std::vector<ncclComm_t> comms(nGPUs);
    std::vector<int> devs(nGPUs);
    std::vector<cudaStream_t> streams(nGPUs);
    
    for (int i = 0; i < nGPUs; i++) {
        devs[i] = i;
    }
    
    Timer timer;
    timer.tic();
    
    // 直接创建7卡通信组
    NCCLCHECK(ncclCommInitAll(comms.data(), nGPUs, devs.data()));
    
    double init_time = timer.toc();
    std::cout << "通信组创建时间: " << std::fixed << std::setprecision(3) 
              << init_time << " ms" << std::endl;
    
    // 创建stream并测试AllReduce
    size_t dataSize = 64 * 1024 * 1024; // 64MB
    std::vector<float*> d_data(nGPUs);
    
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        CUDACHECK(cudaMalloc(&d_data[i], dataSize * sizeof(float)));
        CUDACHECK(cudaMemset(d_data[i], 1, dataSize * sizeof(float)));
    }
    
    // 预热
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce(d_data[i], d_data[i], dataSize, 
                                ncclFloat, ncclSum, comms[i], streams[i]));
    }
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // 正式测试AllReduce
    timer.tic();
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce(d_data[i], d_data[i], dataSize, 
                                ncclFloat, ncclSum, comms[i], streams[i]));
    }
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    double allreduce_time = timer.toc();
    
    double data_gb = (dataSize * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bus_bandwidth = data_gb * 2 * (nGPUs - 1) / nGPUs / (allreduce_time / 1000.0);
    
    std::cout << "AllReduce 时间: " << allreduce_time << " ms" << std::endl;
    std::cout << "数据大小: " << data_gb << " GB" << std::endl;
    std::cout << "算法带宽 (Algbw): " << bus_bandwidth << " GB/s" << std::endl;
    
    // 清理
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_data[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
}

// 测试从8卡通信组split出7卡通信组
void test_split_7gpu_from_8gpu() {
    std::cout << "\n=== Test 2: 从8卡通信组split出7卡通信组 ===" << std::endl;
    
    int nGPUs = 8;
    std::vector<ncclComm_t> comms_8(nGPUs);
    std::vector<ncclComm_t> comms_7(nGPUs);
    std::vector<int> devs(nGPUs);
    std::vector<cudaStream_t> streams(nGPUs);
    
    for (int i = 0; i < nGPUs; i++) {
        devs[i] = i;
    }
    
    Timer timer;
    
    // 先创建8卡通信组
    timer.tic();
    NCCLCHECK(ncclCommInitAll(comms_8.data(), nGPUs, devs.data()));
    double init_8gpu_time = timer.toc();
    std::cout << "8卡通信组创建时间: " << std::fixed << std::setprecision(3) 
              << init_8gpu_time << " ms" << std::endl;
    
    // 从8卡split出7卡（排除第7号卡）
    timer.tic();
    for (int i = 0; i < nGPUs; i++) {
        CUDACHECK(cudaSetDevice(i));
        int color = (i == 7) ? NCCL_SPLIT_NOCOLOR : 0; // 第7号卡不参与
        NCCLCHECK(ncclCommSplit(comms_8[i], color, i, &comms_7[i], NULL));
    }
    double split_time = timer.toc();
    std::cout << "CommSplit 时间: " << split_time << " ms" << std::endl;
    std::cout << "总创建时间 (8卡+split): " << (init_8gpu_time + split_time) << " ms" << std::endl;
    
    // 在7卡通信组上测试AllReduce
    size_t dataSize = 64 * 1024 * 1024; // 64MB
    std::vector<float*> d_data(nGPUs);
    
    for (int i = 0; i < 7; i++) { // 只使用前7张卡
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&streams[i]));
        CUDACHECK(cudaMalloc(&d_data[i], dataSize * sizeof(float)));
        CUDACHECK(cudaMemset(d_data[i], 1, dataSize * sizeof(float)));
    }
    
    // 预热
    for (int i = 0; i < 7; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce(d_data[i], d_data[i], dataSize, 
                                ncclFloat, ncclSum, comms_7[i], streams[i]));
    }
    for (int i = 0; i < 7; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    
    // 正式测试AllReduce
    timer.tic();
    for (int i = 0; i < 7; i++) {
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclAllReduce(d_data[i], d_data[i], dataSize, 
                                ncclFloat, ncclSum, comms_7[i], streams[i]));
    }
    for (int i = 0; i < 7; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
    double allreduce_time = timer.toc();
    
    double data_gb = (dataSize * sizeof(float)) / (1024.0 * 1024.0 * 1024.0);
    double bus_bandwidth = data_gb * 2 * 6 / 7 / (allreduce_time / 1000.0);
    
    std::cout << "AllReduce 时间: " << allreduce_time << " ms" << std::endl;
    std::cout << "数据大小: " << data_gb << " GB" << std::endl;
    std::cout << "算法带宽 (Algbw): " << bus_bandwidth << " GB/s" << std::endl;
    
    // 清理
    for (int i = 0; i < 7; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(d_data[i]));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms_7[i]);
    }
    for (int i = 0; i < nGPUs; i++) {
        ncclCommDestroy(comms_8[i]);
    }
}

int main() {
    std::cout << "NCCL CommSplit 性能测试" << std::endl;
    std::cout << "======================================" << std::endl;
    
    // 检查GPU数量
    int nGPUs;
    CUDACHECK(cudaGetDeviceCount(&nGPUs));
    std::cout << "检测到 " << nGPUs << " 张GPU" << std::endl;
    
    if (nGPUs < 8) {
        std::cerr << "需要至少8张GPU才能运行此测试！" << std::endl;
        return 1;
    }
    
    // 测试1: 直接创建7卡通信组
    test_direct_7gpu_comm(7);
    
    // 测试2: 从8卡split出7卡
    test_split_7gpu_from_8gpu();
    
    std::cout << "\n=== 总结 ===" << std::endl;
    std::cout << "请对比以上两种方式的通信组创建时间和AllReduce性能" << std::endl;
    
    return 0;
}
