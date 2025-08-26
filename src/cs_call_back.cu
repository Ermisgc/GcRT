#include "cs_call_back.h"

namespace GcRT{
    void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data) {
        std::cout << "Callback from stream " << *((int*)data) << std::endl;
    }



}