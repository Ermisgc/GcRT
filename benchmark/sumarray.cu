/**
 * @file sumarray.cpp
 * @brief 通过数组求和的例子测试不同的内存/显存分配形式的性能
 */

// #include <iostream>
#include "cuda_runtime.h"
#include "cstring"
#include <string>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y, float * res) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;  //x坐标的索引
  int stride = blockDim.x * gridDim.x;        //每一个线程执行对应x相同的一列元素的加和
  for (int i = index; i < n; i += stride)
    res[i] = x[i] + y[i];
}

int main(int argc, char ** argv)
{
    int N = 1<<26, TRIALS = 30;
    float total_ms = 0.0f;
        // Launch kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    float *x, *y, *res;
    float *hx, *hy, *hr;
    if(argc < 2){
        cudaMalloc(&x, N * sizeof(float));
        cudaMalloc(&y, N * sizeof(float));
        cudaMalloc(&res, N * sizeof(float));
        hx = (float *)malloc(N * sizeof(float));
        hy = (float *)malloc(N * sizeof(float));
        hr = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++) {
            hx[i] = 1.0f;
            hy[i] = 2.0f;
        }
        cudaMemcpy(x, hx, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y, hy, N * sizeof(float), cudaMemcpyHostToDevice);
        
        while(TRIALS--){
            add<<<numBlocks, blockSize>>>(N, x, y, res);
            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();
        }

        //std::cout << "Unified Memory in " << N << " datas, " << TRIALS <<" tries for malloc, compute and free has an aver time: " << total_ms / TRIALS << "ms" << std::endl;
        cudaFree(x);
        cudaFree(y);
        cudaFree(res); 
        cudaMemcpy(hr, res, N *sizeof(float), cudaMemcpyDeviceToHost);
        free(hx);
        free(hy);
        free(hr);
    }

    else if(std::string(argv[1]) == "unified"){
        cudaMallocManaged(&x, N*sizeof(float));
        cudaMallocManaged(&y, N*sizeof(float));
        cudaMallocManaged(&res, N*sizeof(float));
        // initialize x and y arrays on the host
        for (int i = 0; i < N; i++) {
            x[i] = 1.0f;
            y[i] = 2.0f;
        }

        while(TRIALS--){
            add<<<numBlocks, blockSize>>>(N, x, y, res);
            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();
        }
        cudaFree(x);
        cudaFree(y);
        cudaFree(res);

    } else if(std::string(argv[1]) == "pinned"){
        cudaMalloc(&x, N * sizeof(float));
        cudaMalloc(&y, N * sizeof(float));
        cudaMalloc(&res, N * sizeof(float));
        cudaMallocHost((void**)&hx, N * sizeof(float));
        cudaMallocHost((void**)&hy, N * sizeof(float));
        for (int i = 0; i < N; i++) {
            hx[i] = 1.0f;
            hy[i] = 2.0f;
        }
        cudaMemcpy(x, hx, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(y, hy, N * sizeof(float), cudaMemcpyHostToDevice);
        
        while(TRIALS--){
            add<<<numBlocks, blockSize>>>(N, x, y, res);
            // Wait for GPU to finish before accessing on host
            cudaDeviceSynchronize();
        }

        //std::cout << "Unified Memory in " << N << " datas, " << TRIALS <<" tries for malloc, compute and free has an aver time: " << total_ms / TRIALS << "ms" << std::endl;
        cudaFree(x);
        cudaFree(y);
        cudaFree(res); 
        cudaFreeHost(hx);
        cudaFreeHost(hy);
    }
    return 0;
}