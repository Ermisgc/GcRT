#include <NvInfer.h>
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include <cassert>
#include "inference_meta.h"

namespace GcRT{
    //一个流水线任务的元数据
    typedef void (*Callback)(cudaStream_t, cudaError_t, void *);

    //批处理上下文
    struct BatchContext{
        std::vector<InferenceReq> requests;
        ExecutionContextMeta * context_meta;
        std::vector<void *> input_device_ptrs;
        std::vector<void *> output_device_ptrs;
        cudaEvent_t completion_event;
    };

    //流水线，一条流水线绑定一个流
    class Pipeline{
        cudaStream_t _stream;
        int _priority;

    public:
        Pipeline(int priority = 0);
        ~Pipeline();

        void submit(std::vector<InferenceReq> & requests, ExecutionContextMeta * context_meta);

        int get_priority();

    private:
        //空间分配
        void prepareDeviceMemory(std::shared_ptr<BatchContext> batch_ctx);

        //异步执行拷贝
        void copyInputToDevice(std::shared_ptr<BatchContext> batch_ctx);

        //执行上下文输入输出张量绑定
        void bindExecutionContext(std::shared_ptr<BatchContext> batch_ctx);

        //执行异步推理的操作
        void executeInference(std::shared_ptr<BatchContext> batch_ctx);

        //设定推理完成后的回调
        void setupCompletionCallback(std::shared_ptr<BatchContext> batch_ctx);

        //设定最终拷贝完成后的
        static void setupFinalCallback(std::shared_ptr<BatchContext> batch_ctx, cudaStream_t stream);
    };
}