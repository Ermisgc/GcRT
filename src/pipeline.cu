#include "pipeline.h"
#include "memory/gpu_memory_pool.h"
namespace GcRT{
    Pipeline::Pipeline(int priority = 0){
        this->_priority = priority;
        cudaStreamCreateWithPriority(&_stream, cudaStreamNonBlocking, priority);
    }

    Pipeline::~Pipeline(){
        cudaStreamDestroy(_stream);
    }

    void Pipeline::submit(std::vector<InferenceReq> & requests, ExecutionContextMeta * context_meta){
        auto batch_ctx = std::make_shared<BatchContext>();
        batch_ctx->requests = std::move(requests);
        batch_ctx->context_meta = context_meta;
        cudaEventCreate(&batch_ctx->completion_event);

        prepareDeviceMemory(batch_ctx);

        copyInputToDevice(batch_ctx);

        bindExecutionContext(batch_ctx);

        executeInference(batch_ctx);

        setupCompletionCallback(batch_ctx);
    }

    //空间分配
    void Pipeline::prepareDeviceMemory(std::shared_ptr<BatchContext> batch_ctx){
        ExecutionContextMeta * meta = batch_ctx->context_meta;
        nvinfer1::IExecutionContext * context = meta->ctx;
        for(int i = 0;i < meta->nb_input; ++i){
            meta->input_ptrs[i] = GPUMemoryPool::instance().allocate(meta->input_sizes[i]);
        }

        for(int i = 0;i < meta->nb_output; ++i){
            meta->output_ptrs[i] = GPUMemoryPool::instance().allocate(meta->output_sizes[i]);
        }
    }

    //异步执行拷贝
    void Pipeline::copyInputToDevice(std::shared_ptr<BatchContext> batch_ctx){
        //设置断言，两者应该相同
        assert(batch_ctx->requests.size() == batch_ctx->context_meta->batch_size);
        ExecutionContextMeta * meta = batch_ctx->context_meta;

        int batch_count = 0, batch_size = meta->batch_size;

        //绑定异步拷贝输入输出张量
        for(auto & req: batch_ctx->requests){
            for(int i = 0;i < batch_ctx->context_meta->nb_input; ++i){
                cudaMemcpyAsync(static_cast<char *>(meta->input_ptrs[i]) + batch_count * meta->input_sizes[i] / batch_size, 
                                req.h_input_buffer[i], 
                                req.input_sizes[i], 
                                cudaMemcpyHostToDevice, 
                                _stream);
            }
            batch_count++;
        }
    }

    //执行上下文输入输出张量绑定
    void Pipeline::bindExecutionContext(std::shared_ptr<BatchContext> batch_ctx){
        auto * meta = batch_ctx->context_meta;
        auto * context = batch_ctx->context_meta->ctx;

        for(int i = 0;i < meta->nb_input; ++i){
            context->setInputTensorAddress(meta->input_names[i].c_str(), meta->input_ptrs[i]);
        }
        for(int i = 0;i < meta->nb_output; ++i){
            context->setOutputTensorAddress(meta->output_names[i].c_str(), meta->output_ptrs[i]);
        }

        context->setOptimizationProfileAsync(0, _stream);
    }

    //执行异步推理的操作
    void Pipeline::executeInference(std::shared_ptr<BatchContext> batch_ctx){
        auto * meta = batch_ctx->context_meta;
        auto * context = meta->ctx;
        if(context->enqueueV3(_stream) == cudaSuccess){
            cudaEventRecord(batch_ctx->completion_event, _stream);
        } else {
            //TODO:日志打印输出
            std::cout << "executeInference failed" << std::endl;
        }
    }

    //设定回调
    void Pipeline::setupCompletionCallback(std::shared_ptr<BatchContext> batch_ctx){
        auto callback_wrapper = [](cudaStream_t stream, cudaError_t status, void * userData){
            auto batch_ctx_ptr = static_cast<std::shared_ptr<BatchContext> *>(userData);
            auto batch_ctx = *batch_ctx_ptr;
            delete batch_ctx_ptr;

            if(status != cudaSuccess){
                //TODO:日志打印输出
                std::cout << "callback_wrapper failed" << std::endl;
            }

            auto meta = batch_ctx->context_meta;
            int batch_size = meta->batch_size;

            //将数据拷贝回主机
            for(int i = 0;i < batch_ctx->requests.size(); ++i){
                auto & req = batch_ctx->requests[i];
                for(int j = 0;j < req.h_output_buffer.size(); ++j){
                    cudaMemcpyAsync(req.h_output_buffer[j], 
                                    static_cast<char *>(meta->output_ptrs[j]) + i * meta->output_sizes[j] / batch_size, 
                                    req.output_sizes[j], 
                                    cudaMemcpyDeviceToHost, 
                                    stream);
                }
            }

            Pipeline::setupFinalCallback(batch_ctx, stream);
        };

        auto * userData = new std::shared_ptr<BatchContext>(batch_ctx);
        cudaStreamAddCallback(_stream, callback_wrapper, userData, 0);
    }

    void Pipeline::setupFinalCallback(std::shared_ptr<BatchContext> batch_ctx, cudaStream_t static_stream){
        auto final_callback = [](cudaStream_t stream, cudaError_t status, void * userData){
            auto batch_ctx_ptr = static_cast<std::shared_ptr<BatchContext> *>(userData);
            auto batch_ctx = *batch_ctx_ptr;
            delete batch_ctx_ptr;

            if(status != cudaSuccess){
                //TODO:日志打印输出
                std::cout << "final_callback failed" << std::endl;
            }

            //调用回调函数
            for(int i = 0;i < batch_ctx->requests.size(); ++i){
                auto & req = batch_ctx->requests[i];
                req.call_back(stream, status, req.user_data);
            }


            //释放事件
            cudaEventDestroy(batch_ctx->completion_event);

            //释放内存
            for(int i = 0;i < batch_ctx->context_meta->nb_input; ++i){
                GPUMemoryPool::instance().deallocate(batch_ctx->context_meta->input_ptrs[i]);
            }
            for(int i = 0;i < batch_ctx->context_meta->nb_output; ++i){
                GPUMemoryPool::instance().deallocate(batch_ctx->context_meta->output_ptrs[i]);
            }

            //TODO:归还上下文
        };

        auto * userData = new std::shared_ptr<BatchContext>(batch_ctx);
        cudaStreamAddCallback(static_stream, final_callback, userData, 0);
    }

    int Pipeline::get_priority(){
        return _priority;
    }
}