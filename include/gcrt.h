#include <NvInfer.h>
#include <string>
#include <memory>
#include <thread>
#include <vector>
#include "scheduler.h"
#include "inference_meta.h"

namespace GcRT{
    namespace asio = boost::asio;

    class Scheduler;
    class GPUMemoryPool;   //在memory/gpu_memory_pool.h中实现
    class PinnedMemoryPool;  
    class HttpServer;
    class PipelineManager;
    class ModelImportConfig;
    class ModelSwitchOption;
    
    class GcRT{
    public:
        static GcRT & getInstance();

        void run();

        void addEngine(const std::string & engine_id, const std::string & engine_path, const std::vector<int> & batch_sizes = {1, 2, 4, 8});
    private:
        GcRT();

        std::unique_ptr<HttpServer> _server;
        std::unique_ptr<Scheduler> _engine_scheduler;
        std::shared_ptr<PipelineManager> _pipeline_manager;
    };
}