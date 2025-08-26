#include <NvInfer.h>
#include <string>
#include <memory>
#include <thread>
#include <vector>
#include "engine_scheduler.h"
#include "inference_meta.h"

namespace GcRT{
    namespace asio = boost::asio;

    class InferenceHandle;
    class InferenceRequest;
    class EngineScheduler;
    class GPUMemoryPool;   //在memory/gpu_memory_pool.h中实现
    class PinnedMemoryPool;
    class HttpServer;
    class HttpSession;
    class PipelineManager;
    
    class GcRT{
    public:
        static GcRT & getInstance();

        //管理端接口
        bool loadModel(const std::string & model_id, const ModelImportConfig & config);

        bool unloadModel(const std::string & model_id);

        //客户端提交推理任务，基于future/promise返回异步推理结果
        void submit(const InferenceRequest & req);

        void run();

    private:
        GcRT();

        std::unique_ptr<HttpServer> _server;
        std::unique_ptr<EngineScheduler> _engine_scheduler;
        std::unique_ptr<GPUMemoryPool> _gpu_memory_pool;
        std::unique_ptr<PinnedMemoryPool> _pinned_memory_pool;
        std::unique_ptr<PipelineManager> _pipeline_manager;
    };

}