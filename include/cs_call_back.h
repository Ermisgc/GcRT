/**
 * @file cs_call_back: CallbackData Used for cudaStream_t's Callback
 */

 #pragma once
 #include <NvInfer.h>
 #include "tensor.h"
 #include <atomic>

 namespace GcRT{
    static std::atomic<int> task_id(0);

    struct CallBackData {
        TensorUniquePtr host_output_receiver;
        size_t output_size;
    };
    
    void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data); 
}  //namespace GcRT

