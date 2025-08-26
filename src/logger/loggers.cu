#include "logger/loggers.h"

namespace GcRT{
BuildLogger::BuildLogger(Severity set_severity):logger_serverity(set_severity), nvinfer1::ILogger(){

}

BuildLogger::~BuildLogger(){

}

void BuildLogger::handleMsg(Severity s, const char * msg){
    switch (s)
    {
    case Severity::kERROR:
        std::cout << "[TRT ERROR] "; 
        break;
    case Severity::kWARNING:
        std::cout << "[TRT WARNING] "; 
        break;
    case Severity::kINFO:
        std::cout << "[TRT INFO] ";
        break;
    case Severity::kINTERNAL_ERROR:
        std::cout << "[TRT INTERNAL_ERROR] ";
        break;
    default:
        std::cout << "[TRT VERBOSE] ";
        break;
    }
    std::cout << msg << std::endl;
}


void BuildLogger::log(Severity severity, const char* msg) noexcept{
    if (severity <= logger_serverity) handleMsg(severity, msg);
}

void BuildLogger::handleCudaError(cudaError_t & err) noexcept{
    if(err == cudaSuccess) return;
    std::cout << "[CUDA ERROR] " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << std::endl;
}

void BuildLogger::handleCudaError(cudaError_t && err) noexcept{
    handleCudaError(err);
}

}