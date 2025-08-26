#pragma once
#include <NvInfer.h>
#include <iostream>

#ifndef __rt_deb
#define __rt_deb 
#endif

namespace GcRT
{
    /**
     * @todo 应该搭建一个spdlog的多输出源异步日志系统
     */
    class BuildLogger: public nvinfer1::ILogger
    {
    public:
        BuildLogger() = delete;
        ~BuildLogger();

        explicit BuildLogger(Severity set_severity);

        void log(Severity severity, const char* msg) noexcept override;

        void handleCudaError(cudaError_t & err) noexcept;

        void handleCudaError(cudaError_t && err) noexcept;

    private:
        void handleMsg(Severity Severity, const char * msg);
        Severity logger_serverity;
    };

    /**
     * @name MemoryLogger
     * @brief 显存分配的日志，用来监控日志手动端显存的分配和释放
     * 
     */
    class MemoryLogger: public nvinfer1::ILogger{
    public:
        MemoryLogger() = default;
        MemoryLogger(const char * filename);
        ~MemoryLogger();

        void log(Severity severity, const char* msg) noexcept override{} //并不实现本Log函数

        void MallocLog(cudaError_t & err, size_t size) noexcept;

        void FreeLog(cudaError_t) noexcept;
    private:       
        
    };
}