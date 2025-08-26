#include "engine.h"
#include "pipeline_manager.h"
#include <fstream>

namespace GcRT{
    Engine::Engine(const std::string & model_url, int max_batch_size, PipelineManager * manager): _max_batch_size(max_batch_size), \
        _logger(BuildLogger::Severity::kINFO), _pipeline_manager(manager){
        assert(max_batch_size > 0);
        loadEngine(model_url);
        while(max_batch_size > 0){
            createContexts(max_batch_size);
            max_batch_size /= 2;
        }
        
    }


    Engine::~Engine(){
        while(_ctx_count.load() > 0){
            for(auto & item : _free_list){
                ExecutionContextMeta * context_meta = nullptr;
                while(!item.second.try_dequeue(context_meta)){
                    ExecutionContextMeta * context_meta = nullptr;
                    if(context_meta){
                        delete context_meta;
                        _ctx_count.fetch_sub(1);
                    }
                }
            }

            if(_ctx_count.load() == 0) break;

            {
                std::unique_lock<std::mutex> locker(_mtx);
                _ret_cv.wait(locker);
            }
        }

    }

    void Engine::addRequest(Request * req){
        _req_list.enqueue(req);
        _req_size++;
    }

    void Engine::dynamicBatchHandle(){
        //动态批处理，首先是看最上层的执行上下文有没有ExecutionMeta
        int batch_size = _max_batch_size;
        int current_req_size = _req_size.load();  //提前锁当前的请求量
        
        while(batch_size > 0){
            while(batch_size > 0 && current_req_size < batch_size){
                batch_size /= 2;
            }

            ExecutionContextMeta * meta;
            while(batch_size > 0 && !_free_list[batch_size].try_dequeue(meta)){
                //push不出来，那么batch_size要减小了
                batch_size /= 2;
            }

            if(!meta) break;  //此时还没有扔出来，就只能是batch_size == 0了，直接break;

            //开始组装请求
            _req_size.fetch_sub(batch_size);
            current_req_size -= batch_size;

            std::vector<InferenceReq *> requests;
            for(int i = 0;i < batch_size; ++i){
                Request * req;
                if(_req_list.try_dequeue(req)){
                    InferenceReq * infer_req = new InferenceReq;
                    //TODO:具体参数组装
                    requests.push_back(infer_req);
                    
                } else {
                    std::cout << "Invalid Inner Code Error" << std::endl;
                }
            }
            
            _pipeline_manager->submit(std::move(requests), meta);
        }
    }
    
    void Engine::returnContext(ExecutionContextMeta * context_meta){
        _free_list[context_meta->batch_size].enqueue(context_meta);
        _ret_cv.notify_one();
    }

    size_t Engine::getRequestQueueSize() const{
        return _req_size;
    }

    std::string Engine::getEngineInfo() const{
        return "";
    }

    std::vector<int> Engine::getAvailableBatchSizes() const{
        std::vector<int> batch_sizes;
        for(auto & item : _free_list){
            batch_sizes.push_back(item.first);
        }
        return batch_sizes;
    }
    
    bool Engine::loadEngine(const std::string & engine_path){
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            std::cerr << "Failed to open engine file: " << engine_path << std::endl;
            return false;
        }
        
        // 读取引擎文件内容
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(size);
        engine_file.read(engine_data.data(), size);
        engine_file.close();
        
        // 创建运行时并反序列化引擎
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(_logger);
        if (!runtime) {
            std::cerr << "Failed to create TensorRT runtime" << std::endl;
            return false;
        }
        
        _engine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data.data(), size));
        delete runtime;
        
        if (!_engine) {
            std::cerr << "Failed to deserialize engine" << std::endl;
            return false;
        }
        
        return true;
    }

    bool Engine::createContexts(int batch_size){
        nvinfer1::IExecutionContext* context = _engine->createExecutionContext();
        if (!context) {
            std::cerr << "Failed to create execution context for batch size: " << batch_size << std::endl;
            return false;
        }
        
        // 创建执行上下文元数据
        auto context_meta = new ExecutionContextMeta;
        context_meta->ctx = context;
        context_meta->batch_size = batch_size;
        
        // 设置输入输出信息
        context_meta->nb_input = 0;
        context_meta->nb_output = 0;
        for (int i = 0; i < context_meta->nb_input; ++i) {
            const char * name = _engine->getIOTensorName(i);
            if (_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                context_meta->input_names.push_back(name);
                
                // 获取绑定维度并设置批处理大小
                nvinfer1::Dims dims = context->getTensorShape(name);
                dims.d[0] = batch_size; // 假设批处理维度是第一个维度
                
                // 计算输入大小
                size_t element_size = 0;
                switch (_engine->getTensorDataType(name)) {
                    case nvinfer1::DataType::kFLOAT: element_size = 4; break;
                    case nvinfer1::DataType::kHALF: element_size = 2; break;
                    case nvinfer1::DataType::kINT8: element_size = 1; break;
                    case nvinfer1::DataType::kINT32: element_size = 4; break;
                    default: element_size = 4;
                }
                
                size_t size = element_size;
                for (int k = 0; k < dims.nbDims; ++k) {
                    size *= dims.d[k];
                }
                
                context_meta->input_sizes.push_back(size);
                context_meta->nb_input ++;
            } else if( _engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT ){
                context_meta->output_names.push_back(name);
                
                // 计算输出大小
                nvinfer1::Dims dims = context->getTensorShape(name);
                dims.d[0] = batch_size;
                size_t element_size = 0;
                switch (_engine->getTensorDataType(name)) {
                    case nvinfer1::DataType::kFLOAT: element_size = 4; break;
                    case nvinfer1::DataType::kHALF: element_size = 2; break;
                    case nvinfer1::DataType::kINT8: element_size = 1; break;
                    case nvinfer1::DataType::kINT32: element_size = 4; break;
                    default: element_size = 4;
                }
                
                size_t size = element_size;
                for (int k = 0; k < dims.nbDims; ++k) {
                    size *= dims.d[k];
                }
                
                context_meta->output_sizes.push_back(size);
                context_meta->nb_output ++;
            }
        }
        
        // 将执行上下文添加到空闲队列
        context_meta->input_ptrs.resize(context_meta->nb_input);
        context_meta->output_ptrs.resize(context_meta->nb_output);
        _free_list[batch_size].enqueue(context_meta);
        
        return true;
    }
}