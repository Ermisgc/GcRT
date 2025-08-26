#include "pipeline.h"
#include <mutex>

namespace GcRT{
    //TODO:负载的动态更新机制还未完成
    class PipelineManager{
        //pipeline表
        std::map<int, std::unique_ptr<Pipeline>> _pipelines;
        
        //互斥锁
        mutable std::mutex _mtx;

        //负载均衡计算器
        std::map<int, std::atomic<int>> _pipeline_loads;

        //优先级映射函数:InferenceReq->int
        std::function<int(const InferenceReq &)> _priority_mapper;

        //int -> int
        int mapRequestPriorityToPipeline(int request_priority) const;

        //选择合适的Pipeline
        Pipeline * selectPipeline(const std::vector<InferenceReq> & req);

    public:
        PipelineManager(int cudaStream_count);
        PipelineManager(const std::vector<int> & pipeline_priorities = {-1, 0, 1, 2});
        ~PipelineManager();

        void submit(std::vector<InferenceReq> & requests, ExecutionContextMeta * context_meta);

        //获得指定优先级的Pipeline的当前负载
        int getPipelineLoad(int priority) const;

        std::map<int, int> getAllPipelineLoads() const;

        void setPriorityMapper(std::function<int(const InferenceReq &)> priority_mapper);

        int get_pipeline_count();        
    };
}