#include "inferencer.h"
#include <NvInferRuntime.h>
#include "model_importer.h"
#include "memory/gpu_memory_pool.h"
#include <fstream>

namespace GcRT{

Inferencer::Inferencer(std::shared_ptr<nvinfer1::ICudaEngine> set_engine): context(std::unique_ptr<nvinfer1::IExecutionContext>(set_engine->createExecutionContext())){
    //创建执行上下文
    // context->setInputShape(set_engine->getIOTensorName(0), nvinfer1::Dims);
    logger = std::make_unique<InferenceLogger>(Level::kINFO);
    if(!context){
        logger->log(Level::kERROR, "Fail to create execution context");
        return;
    }

    setUpExecutionContext();
}

Inferencer::Inferencer(const std::string & set_engine){
    logger = std::make_unique<InferenceLogger>(Level::kINFO);
    std::unique_ptr<ModelImporter> importer = std::make_unique<ModelImporter>(set_engine);
    if(auto ret = importer->Engine(); ret.has_value()){
        engine = ret.value();
    } else{
        logger->log(Level::kERROR, "Fail to create engine.");
        return;
    }

    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    setUpExecutionContext();
}

Inferencer::~Inferencer(){
    //确保执行上下文先于engine释放
    if(context){
        context.reset();  //这里要用reset而不是release，因为release并不会释放对象
    }

    cudaStreamDestroy(cuda_stream);
}

inline void Inferencer::setUpExecutionContext(){
    for(int i = 0;i < engine->getNbIOTensors(); ++i){
        const char * name = engine->getIOTensorName(i);
        if(engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT){
            nvinfer1::Dims input_tensor_i = context->getTensorShape(name);
            TensorUniquePtr p = createCudaTensor(engine->getTensorDataType(name), input_tensor_i);

            //将GPU上面的内存空间地址暴露给执行上下文
            context->setTensorAddress(name, p->data().get());
            input_tensors.emplace_back(std::move(p));
        } else if(engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT){
            //当前只假设有一个输出tensor
            if(output_tensor){
                logger->log(Level::kERROR, "Model should have sonly one output");
                exit(EXIT_FAILURE);
            }  

            //检查Dims是否有效
            nvinfer1::Dims o_d = context->getTensorShape(name);
            if(o_d.nbDims == -1){
                if(o_d.d[0] == 0) logger->log(Level::kERROR, (std::string("invalid dims of") + name).c_str());
                else logger->log(Level::kERROR, (std::string("unknown rank of") + name).c_str());
                exit(EXIT_FAILURE);
            }

            TensorUniquePtr p = createCudaTensor(engine->getTensorDataType(name), o_d);
            output_tensor = std::move(p);
            context->setTensorAddress(name, output_tensor->data().get());
        } else { //nvinfer1::TensorIOMode::kNone
            logger->log(Level::kWARNING, "An unrecognizable tensor exist.");
        }
    }

    logger->log(Level::kINFO, "Inferencer created.");

    // create cuda stream
    logger->handleCudaError(cudaStreamCreate(&cuda_stream));
    // cuda_stream = std::unique_ptr<cudaStream_t, CudaStreamDestructor>(temp_cuda_str);
}

//TODO:需要编辑响应，先针对图像分类和语言处理设置响应
bool Inferencer::infer(const InferenceReq & req, std::vector<std::vector<int>> & outputData){
    const std::string enginePath = getEnginePath(req._model_config);

    auto itr = _engineMap.begin();
    //获取引擎资源
    {
        std::lock_guard<std::mutex> locker(_map_mtx);
        itr = _engineMap.find(enginePath);
        if(itr == _engineMap.end()){  //说明engine并未实现加载，需要由管理端实现模型的加载
            std::cerr << "Model not loaded: " << enginePath << std::endl;
            return false;  
        }
    }
    
    EngineMeta & infer_meta = itr->second;

    //准备输入输出
    std::vector<void *> gpuInputBuffers;
    std::vector<size_t> inputSizes;
    std::vector<std::string> inputNames;

    for(const auto & input: req._inputs){
        if(!input.data.empty()){
            std::vector<uint8_t> binaryData = base64Decoding(input.data);
            size_t dataSize = binaryData.size();
            //绑定到infer_meta的Cuda流
            void * gpuInput = GPUMemoryPool::instance().allocate(dataSize, infer_meta.stream);

            cudaMemcpyAsync(gpuInput, binaryData.data(), dataSize, cudaMemcpyHostToDevice, infer_meta.stream);
            
            gpuInputBuffers.push_back(gpuInput);
            inputSizes.push_back(dataSize);
            inputNames.push_back(input.name);
        } else if(!input.data_ref.uri.empty()){
            //TODO:外部数据的实现
        } else {
            std::cerr << "No Invalid Input" << std::endl;
        }
    }

    std::vector<void *> gpuOutputBuffers;
    std::vector<size_t> outputSizes;
    std::vector<std::string> outputNames;
    std::vector<std::string> postprocessing;

    for(const auto & output : req._outputs){
        outputNames.push_back(output.name);
        nvinfer1::Dims o_d = context->getTensorShape(output.name.c_str());
        size_t outputSize = 1;
        for(int i = 0; i < o_d.nbDims; ++i){
            outputSize *= o_d.d[i];
        }

        void * gpuOutput = GPUMemoryPool::instance().allocate(outputSize, infer_meta.stream);
        gpuOutputBuffers.push_back(gpuOutput);
        outputSizes.push_back(outputSize);
    }

    //设置输入/输出绑定
    for(int i = 0;i < inputNames.size(); ++i){
        infer_meta.context->setTensorAddress(inputNames[i].c_str(), gpuInputBuffers[i]);
    }
    for(int i = 0;i < outputNames.size(); ++i){
        infer_meta.context->setTensorAddress(outputNames[i].c_str(), gpuOutputBuffers[i]);
    }


    //执行异步推理
    if(!infer_meta.context->enqueueV3(cuda_stream)){
        logger->log(Level::kERROR, "The inference messions are not enqueued to cuda stream");
        return false;
    }

    cudaStreamAddCallback(infer_meta.stream, [](cudaStream_t stream, cudaError_t status, void* data){}, nullptr, 0);
    
    //TODO:在Callback中集成下面几步
    //推理结果异步拷贝到CPU
    //清理GPU显存
    //后处理process
    //通知对应的HttpSession进行异步发送，或者通过future/promise机制，填写future相应内容
}

std::string Inferencer::getEnginePath(const ModelConfig & config) {
    return "models/" + config.name + "_v" + config.version + ".engine";
}

bool Inferencer::loadModel(const ModelConfig & config){
    std::string enginePath = getEnginePath(config);

    {
        std::lock_guard<std::mutex> map_locker(_map_mtx);
        if(_engineMap.find(enginePath) != _engineMap.end()) return true;
    }
    
    std::ifstream engineFile(enginePath, std::ios::binary);
    if(!engineFile) {
        std::cerr << "Error opening engine file: " << enginePath << std::endl;
        return false;
    }

    std::string modelData = "";
    while (engineFile.peek() != EOF) { // 使用fin.peek()防止文件读取时无限循环
        std::stringstream buffer;
        buffer << engineFile.rdbuf();
        modelData.append(buffer.str());
    }

    EngineMeta em;
    em.runtime = nvinfer1::createInferRuntime(*logger);
    if (!em.runtime) return false;

    em.engine = em.runtime->deserializeCudaEngine(modelData.data(), modelData.size());
    if(!em.engine) {
        delete em.runtime;
        return false;
    }

    em.context = em.engine->createExecutionContext();
    if (!context) {
        delete em.engine;
        delete em.runtime;
        return false;
    }

    // 创建CUDA流
    logger->handleCudaError(cudaStreamCreate(&em.stream));

    {
        std::lock_guard<std::mutex> map_locker(_map_mtx);
        _engineMap[enginePath] = std::move(em);
    }
    
    return true;
}

bool Inferencer::unloadModel(const std::string model_name){
    
}

}