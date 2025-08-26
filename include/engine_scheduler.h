#include <Nvinfer.h>
#include "data_structure/concurrentqueue.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace GcRT{
    class InferenceRequest;
    class Pipeline;
    class ModelManager;

    struct ModelImportConfig{
        std::vector<std::pair<int, int>> batch_args;
        const std::string model_path;
    };

    struct IExecutionContextMeta{
        int batch_size;
        nvinfer1::IExecutionContext * ec;
    };

    struct IExecutionContextMetaCompare{
        bool operator()(const IExecutionContextMeta & a, const IExecutionContextMeta & b){
            return a.batch_size > b.batch_size;
        }
    };

    struct EngineMeta{
        nvinfer1::ICudaEngine * engine;
        std::mutex eng_mtx;
        std::multiset<IExecutionContextMeta> ctx_meta_set;
    };

    class EngineScheduler{
        std::map<std::string, moodycamel::ConcurrentQueue<InferenceRequest>> _req_list;
        std::map<std::string, EngineMeta> _engine_meta_list;        
        std::thread _sched_thread;
        std::atomic<bool> _stop_flag = false;

    public:
        void submit(InferenceRequest && req);

        void load_engine(const std::string & model_name, nvinfer1::ICudaEngine * engine);

        void load_engine(const std::string & model_id, const ModelImportConfig & config);

    private:
        void work();
    };
}