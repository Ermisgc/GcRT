#pragma once
#include <NvInfer.h>
#include <memory>
#include <string>
#include <mutex>
#include "tensor.h"
#include <optional>
#include "logger/loggers.h"
#include <vector>
#include "cs_call_back.h"
#include "inference_meta.h"

namespace GcRT{
//自定义析构的仿函数
struct CudaStreamDestructor{
public:
    void operator()(cudaStream_t *_Ptr) const noexcept
    {
        cudaStreamDestroy(*_Ptr);
    }
};

class Inferencer{
    using InferenceLogger = BuildLogger;
    using Level = InferenceLogger::Severity;
    
public:
    Inferencer() = delete;
    Inferencer(const Inferencer & ) = delete;
    Inferencer(Inferencer &&) = delete;
    Inferencer operator=(const Inferencer &) = delete;
    Inferencer operator=(Inferencer &&) = delete;

    explicit Inferencer(std::shared_ptr<nvinfer1::ICudaEngine> set_engine);

    Inferencer(const std::string & model_file);

    ~Inferencer();

    //TODO:模型的加载、模型库的设计
    bool loadModel(const ModelConfig & config);

    bool unloadModel(const std::string model_name);

    //TOOD:执行推理，根据req进行推理，然后将结果保存在outputData当中
    bool infer(const InferenceReq & req, std::vector<std::vector<int>> & outputData);

    template<typename ... InputPtr>
    std::optional<TensorUniquePtr> inference(InputPtr ... inputs) noexcept{
        size_t base_index = 0;
        bool valid = true;
        //折叠表达式，进行指针检查、尺寸检查、类型检查，以及设置inputs
        ([&](InputPtr & input){
            if(!valid) return;
            if(!isTensorUniquePtr_v<InputPtr>) {
                logger->log(Level::kERROR, "Input should be a unique_ptr of Tensor");
                valid = false;
                return;
            }

            //传入指针为空，直接返回
            if(!input){
                logger->log(Level::kERROR, "Input tensor is empty");
                valid = false;
                return;
            }

            //检查数据类型是否正确
            if(input->type() != input_tensors[base_index]->type()){
                logger->log(Level::kERROR, "Input tensor datatype does not match");
                valid = false;
                return;
            }

            //检查Tensor大小是否正确
            if(input->bytes() != input_tensors[base_index]->bytes()){
                logger->log(Level::kERROR, "Input tensor size does not match");
                valid = false;
                return;
            }

            //将input拷贝到执行上下文当中
            logger->handleCudaError(input_tensors[base_index]->copiedFromHost(*input));
            base_index++;
        } (inputs), ...);    

        if(!valid) return std::nullopt; 
        if(!context->enqueueV3(cuda_stream)){
            logger->log(Level::kERROR, "The inference messions are not enqueued to cuda stream");
            return std::nullopt;
        }

        int stream_id = 1;
        cudaStreamAddCallback(cuda_stream, my_callback, (void*)&stream_id, 0);
        
        logger->handleCudaError(cudaStreamSynchronize(cuda_stream));
        auto ret = createTensorByType(output_tensor->type(), output_tensor->dims());
        logger->handleCudaError(output_tensor->cupyToHost(*ret));
        return ret;
    }

private:
    std::unique_ptr<InferenceLogger> logger;

    struct EngineMeta{
        nvinfer1::ICudaEngine * engine = nullptr;
        nvinfer1::IExecutionContext* context = nullptr;
        nvinfer1::IRuntime * runtime = nullptr;
        cudaStream_t stream = nullptr;
        //TODO:思考：是否有必要增加流同步事件？是否有必要分配到多个流
    };

    std::unordered_map<std::string, EngineMeta> _engineMap;  //模型名字与对应的引擎的映射
    std::mutex _map_mtx;   //用来保护map的互斥锁

    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::vector<TensorUniquePtr> input_tensors;  //这里的input_tensors应该让cudaFree
    // std::unique_ptr<cudaStream_t, CudaStreamDestructor> cuda_stream;
    cudaStream_t cuda_stream;
    TensorUniquePtr output_tensor;  //output_tensors也应该由cudaFree

    size_t m_batch_size = 1;

    //半字节变量怎么存储呢？
    // constexpr static uint8_t sizeOfDataType[11] = {4, 2, 1, 4, 1, 1, 1, 2, 8, 1, 1};
    void setUpExecutionContext();

    std::string getEnginePath(const ModelConfig & config);
};  
}