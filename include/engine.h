#include <NvInfer.h>
#include "loggers.h"
#include "data_structure/concurrentqueue.h"
#include "inference_meta.h"

namespace GcRT{
    class PipelineManager;

    class Engine{
        friend class Scheduler;

        std::shared_ptr<nvinfer1::ICudaEngine> _engine;
        std::map<int, moodycamel::ConcurrentQueue<ExecutionContextMeta *>> _free_list;

        moodycamel::ConcurrentQueue<Request *> _req_list;
        std::atomic<int> _req_size; 
        std::atomic<int> _ctx_count;
        std::mutex _mtx;
        std::condition_variable _ret_cv;
        int _max_batch_size;
        BuildLogger _logger;

        PipelineManager * _pipeline_manager = nullptr;

    public:
        Engine(const std::string & model_url, int max_batch_size, PipelineManager * manager);
        ~Engine();

        void addRequest(Request *);

        // bool getBatch(int batch_size, std::vector<InferenceReq>& requests, ExecutionContextMeta * & context_meta);
        void dynamicBatchHandle();
        
        void returnContext(ExecutionContextMeta * context_meta);

        size_t getRequestQueueSize() const;

        std::string getEngineInfo() const;

        std::vector<int> getAvailableBatchSizes() const;
        
    private:
        bool loadEngine(const std::string & engine_path);

        bool createContexts(int batch_size);
    };
}