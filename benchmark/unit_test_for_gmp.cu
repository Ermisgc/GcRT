#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include "memory/gpu_memory_pool.h"
using namespace std;
GcRT::GPUMemoryPool & pool = GcRT::GPUMemoryPool::instance(); 
cudaMemPool_t cuda_pool;

bool unit_test_ready = true;

void BasicAllocation() {
    void *ptr = pool.allocate(1024);
    if(ptr == nullptr) {
        cerr << "[-] BasicAllocation" << endl;
        unit_test_ready = false;
    }
    else {
        pool.deallocate(ptr);
        cout << "[+] BasicAllocation" << endl;
    }
}

void DataIntegrity() {
    try{
        int * ptr = static_cast<int*>(pool.allocate(1024 * 4));
        vector<int> h_array(1024, 42);
        
        cudaMemcpy(ptr, h_array.data(), 1024 * 4, cudaMemcpyHostToDevice);
        h_array.resize(1024, 0);
        cudaMemcpy(h_array.data(), ptr, 1024 * 4, cudaMemcpyDeviceToHost);

        if(std::all_of(h_array.begin(), h_array.end(), [](int x) -> bool{return x == 42;}));
        pool.deallocate(ptr);
        cout << "[+] DataIntegrity" << endl;
    } catch (const std::exception & e){
        cout << "[-] DataIntegrity: " << e.what() << endl;
        unit_test_ready = false;
    }

}

void ThreadSafety() {
    auto task = [&](int size) {
        try{
            for (int i = 0; i < 2048; ++i) {  //可能发生死锁
                void* ptr = pool.allocate(size);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                pool.deallocate(ptr);
            }
            cout << "[+] ThreadSafety" << endl;
        } catch (const exception & e){
            cout << "[-] ThreadSafety: " << e.what() << endl;
            unit_test_ready = false;
        }
    };

    std::thread t1(task, 128);
    std::thread t2(task, 256);
    t1.join(); t2.join();  // 不应崩溃或死锁
}

void test_latency(bool use_pool, size_t size, int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        void *ptr;
        if(use_pool){
            ptr = pool.allocate(size);
        } else {
            cudaMalloc(&ptr, size);
        }
        // void* ptr = use_pool ? : 
        if (use_pool) pool.deallocate(ptr);
        else cudaFree(ptr);
        // if(i % 1000 ==  0) cout << "done: " << i << endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = end - start;
    if(!use_pool){
        cout << "cudaMalloc Latency: " << duration.count() / 10000.0 << "us/op" << endl;
    } else {
        cout << "Pool Latency: " << duration.count() / 10000.0 << "us/op" << endl;
    }
}

void test_latency_2(bool use_pool, size_t size, int iterations) {
    // 同步设备确保初始状态
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        void *ptr;
        if(use_pool){
            ptr = pool.allocate(size * 1);
        } else {
            cudaMalloc(&ptr, size * 1);
        }
        
        if (use_pool) pool.deallocate(ptr);
        else cudaFree(ptr);
    }
    
    // 确保所有CUDA操作完成
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double avg_latency_us = static_cast<double>(duration_ns.count()) / (iterations * 1'000.0);
    if(use_pool){
        cout << "Pool Latency: " << avg_latency_us << " us/op" << endl;
    } else {
        cout << "cudaMalloc Latency: " << avg_latency_us << " us/op" << endl;
    }
}


void test_concurrency(int thread_count) {
    int iterator = 1000;
    auto worker = [&](int id) {
        std::vector<void*> local_ptrs;
        for (int i = 0; i < iterator; ++i) {
            size_t size = 256 * (1 + (id + i) % 32);  // 不同线程分配不同大小
            void* ptr = pool.allocate(size);
            local_ptrs.push_back(ptr);
        }
        for (void* ptr : local_ptrs) pool.deallocate(ptr);
    };

    std::vector<std::thread> threads;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < thread_count; ++i) threads.emplace_back(worker, i);
    for (auto& t : threads) t.join();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = (end - start).count() / (iterator * 1000.0);
    printf("Concurrency (%d threads): Total Finish Time: %.3f us/op\n", thread_count, duration);
}



int main(){
    // cudaMemPoolCreate();
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));  //准备一秒的预热时间

    void * dummy;
    cudaMalloc(&dummy, 1);
    cudaFree(dummy);
    cudaDeviceSynchronize();

    std::cout << ("Test For Small Memory allocate/deallocate") << std::endl;
    for(int i = 6; i < 21; ++i){
        cout << "Malloc Size = " << (1 << i) << endl;

        test_latency_2(true, (1 << i), 10000);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));  //停顿一秒，防止互相干扰
        test_latency_2(false, (1 << i), 10000);
        cout << "---------------------------" << endl;
    }

    std::cout << "Test For Concurrency" << std::endl;
    vector<int> threads = {1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72};
    for(int i = 0; i < threads.size(); i ++){
        test_concurrency(threads[i]);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));  //停顿一秒，防止互相干扰
    }
    cout << "---------------------------" << endl;

    return 0;
}