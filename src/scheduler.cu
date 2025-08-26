#include "scheduler.h"
#include "pipeline_manager.h"
#include "engine.h"

namespace GcRT{
    Scheduler::Scheduler(std::shared_ptr<PipelineManager> manager){
        _pipeline_manager = manager;
    }

    Scheduler::~Scheduler(){
        stop();
    }

    bool Scheduler::addEngine(const std::string & engine_id, const std::string & model_path, const std::vector<int> & batch_sizes = {1, 2, 4, 8}){
        std::lock_guard<std::mutex> lock(_mtx);
        
        if(_engines.find(engine_id) != _engines.end()){
            std::cerr << "Engine with ID" << engine_id << " already exists" << std::endl;
            return false;
        }

        try{
            auto engine = std::make_shared<Engine>(model_path, batch_sizes);
            _engines[engine_id] = engine;
            std::cout << "Engine " << engine_id << " added successfully" << std::endl;
            return true;
        } catch (const std::exception & e){
            std::cerr << "Fail to add engine " << engine_id << ": " << e.what() << std::endl;
            return false;
        }
    }

    bool Scheduler::removeEngine(const std::string & engine_id){
        std::lock_guard<std::mutex> lock(_mtx);

        auto it = _engines.find(engine_id);
        if (it == _engines.end()) {
            std::cerr << "Engine with ID " << engine_id << " not found" << std::endl;
            return false;
        }
        
        // 移除引擎
        _engines.erase(it);
        std::cout << "Engine " << engine_id << " removed successfully" << std::endl;
        return true;
    }

    void Scheduler::submitInference(const std::string & engine_id, Request * req){
        std::lock_guard<std::mutex> locker(_mtx);
        auto it = _engines.find(engine_id);
        if (it == _engines.end()) {
            std::cerr << "Engine with ID " << engine_id << " not found" << std::endl;
            return;
        } else {
            it->second->addRequest(req);
            _queue_cv.notify_one();
        }
    }

    void Scheduler::submitManagement(ManagementRequest * op){
        _request_queue.enqueue(op);
    }

    void Scheduler::start(){
        if(_running) return;
        _running = true;
        _worker_thread = std::thread(&Scheduler::worker, this);
    } 

    void Scheduler::stop(){
        if(!_running) return;
        _running = false;
        if(_worker_thread.joinable()) _worker_thread.join();
    }

    void Scheduler::worker(){
        while(_running){
            std::unique_lock<std::mutex> locker(_mtx);
            _queue_cv.wait(locker);

            if(!_running) break;

            //对每一个Engine的队列，进行请求处理
            for(auto itr = _engines.begin(); itr != _engines.end(); ++itr){
                itr->second->dynamicBatchHandle();
            }

            //处理管理层请求
            ManagementRequest * mr = nullptr;
            while(_request_queue.try_dequeue(mr)){
                handleManagementRequest(mr);
            }
        }
    }

    void Scheduler::handleManagementRequest(ManagementRequest * mr){
        switch (mr->op)
        {
        case ManagementOp::LOAD_MODEL:
            addEngine(mr->model_id, mr->model_path);
            break;
        case ManagementOp::UNLOAD_MODEL:
            removeEngine(mr->model_id);
            break;
        case ManagementOp::UPDATE_CONFIG:
            //TODO:模型参数更新
            break;
        case ManagementOp::SWITCH_MODEL:
            //TODO:模型热切换
            //等待上一个引擎的内容执行完，这段时间请求应该处于积压状态

            //执行切换
            //removeEngine(mr->model_id);
            //addEngine(mr->model_id, mr->model_path);
            break;
        default:
            break;
        }
    }

}