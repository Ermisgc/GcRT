#include <Nvinfer.h>
#include "data_structure/concurrentqueue.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include "inference_meta.h"

namespace GcRT{
    class Engine;
    class PipelineManager;

    class Scheduler{
        std::unordered_map<std::string, std::shared_ptr<Engine>> _engines;
        std::thread _worker_thread;
        std::atomic<bool> _destroying;

        PipelineManager * _pipeline_manager;

    public:
        Scheduler(PipelineManager * manager);
        ~Scheduler();

        bool addEngine(const std::string & engine_id, const std::string & model_path, const std::vector<int> & batch_sizes = {1, 2, 4, 8});

        bool removeEngine(const std::string & engine_id);;

        void submitInference(const std::string & engine_id, const Request * req);

        void submitManagement(ManagementOp op, const std::string & engine_id, const std::string & model_path, std::function<void(bool)> callback);

        void start(); 

        void stop();
    
    private:
        void worker();  //工作线程
    };
}