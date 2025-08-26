#include "tensor.h"

namespace GcRT{
    TensorUniquePtr createTensorByType(nvinfer1::DataType type, nvinfer1::Dims dims){
        switch (type)
        {
#define case_and_create(type, T) \
        case nvinfer1::DataType::type: \
            return createTensor<T>(dims);

        case_and_create(kFLOAT, float);
        case_and_create(kHALF, __half);
        case_and_create(kINT8, int8_t);
        case_and_create(kUINT8, uint8_t);
        case_and_create(kINT32, int32_t);
        case_and_create(kINT64, int64_t);
        case_and_create(kBOOL, bool);
#undef case_and_create
        default:
            return nullptr;
            break;
        }
    }

    TensorUniquePtr createCudaTensor(nvinfer1::DataType type, nvinfer1::Dims dims){
        return CudaTensor::from_dims(type, dims);
    }   
}