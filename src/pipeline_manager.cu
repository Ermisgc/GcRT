#include "pipeline_manager.h"

namespace GcRT{
    PipelineManager::PipelineManager(int cudaStream_count) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        if (prop.streamPrioritiesSupported == 0) {
            //"This device does not support priority streams";
            for(int i = 0;i < cudaStream_count;i++){
                _pipelines.insert({i, std::make_unique<Pipeline>()});
            }
        } else {
            int priority_low, priority_high;
            cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
            int priority_step = (priority_high - priority_low) / cudaStream_count;
            for(int i = 0;i < cudaStream_count;i++){
                _pipelines.insert({i, std::make_unique<Pipeline>(priority_low + i * priority_step)});
            }
        }
    }

    PipelineManager::PipelineManager(const std::vector<int> & pipeline_priorities = {-1, 0, 1, 2}){
        for(int i = 0;i < pipeline_priorities.size();i++){
            _pipelines.insert({i, std::make_unique<Pipeline>(pipeline_priorities[i])});
        }
    }

    PipelineManager::~PipelineManager(){
        //这里用了unique_ptr，不需要手动析构
    }

    void PipelineManager::submit(std::vector<InferenceReq> & requests, ExecutionContextMeta * context_meta){
        if(requests.empty()){
            return;
        }

        //选择合适的Pipeline
        Pipeline * selected_pipeline = selectPipeline(requests);

        if(!selected_pipeline){
            //没有合适的Pipeline，处理错误
            for(auto & req: requests){
                req.call_back(nullptr, cudaErrorInvalidValue, req.user_data);
            }
            return;
        }

        int pipeline_priority = selected_pipeline->get_priority();
        _pipeline_loads[pipeline_priority]++;

        selected_pipeline->submit(requests, context_meta);
    }

    //获得指定优先级的Pipeline的当前负载
    int PipelineManager::getPipelineLoad(int priority) const{
        auto it = _pipeline_loads.find(priority);
        if (it != _pipeline_loads.end()) {
            return it->second.load();
        }
        return -1; // 表示未找到指定优先级的 Pipeline
    }


    Pipeline * PipelineManager::selectPipeline(const std::vector<InferenceReq> & requests){
        int request_priority = _priority_mapper(requests[0]);
        int target_pipeline_priority = mapRequestPriorityToPipeline(request_priority);

        //有优先级就先根据优先级选
        {
            std::lock_guard<std::mutex> locker(_mtx);
            if(auto itr = _pipelines.find(target_pipeline_priority) ; itr != _pipelines.end()){
                return itr->second.get();
            }
        }

        int min_load = INT_MAX;
        Pipeline* selected_pipeline = nullptr;
        
        for (auto& pair : _pipelines) {
            int load = _pipeline_loads[pair.first];
            if (load < min_load) {
                min_load = load;
                selected_pipeline = pair.second.get();
            }
        }

        return selected_pipeline;
    }

    int PipelineManager::mapRequestPriorityToPipeline(int request_priority) const{
        int closest_priority = INT_MAX;
        int min_diff = INT_MAX;

        for (const auto& pair : _pipelines) {
            int diff = std::abs(pair.first - request_priority);
            if (diff < min_diff) {
                min_diff = diff;
                closest_priority = pair.first;
            }
        }
        
        return closest_priority;        
    }

    //获得所有Pipeline的当前负载
    std::map<int, int> PipelineManager::getAllPipelineLoads() const{
        std::map<int, int> loads;
        for(auto & pipeline : _pipeline_loads){
            loads[pipeline.first] = pipeline.second.load();
        }
        return loads;
    }

    void PipelineManager::setPriorityMapper(std::function<int(const InferenceReq &)> priority_mapper){
        _priority_mapper = priority_mapper;
    }

    int PipelineManager::get_pipeline_count(){
        return _pipelines.size();
    }       


}