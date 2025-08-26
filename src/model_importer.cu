#include "model_importer.h"
#include <NvOnnxParser.h>
#include <fstream>

namespace GcRT{
ModelImporter::ModelImporter(const std::string & imported_model, nvinfer1::NetworkDefinitionCreationFlags flag){
    logger = std::make_unique<BuildLogger>(Level::kWARNING);
    builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*logger));
    network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    //构建配置
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    //用NvOnnxParser导入onnx模型
    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *logger));
    parser->parseFromFile(imported_model.c_str(), static_cast<int32_t>(Level::kINFO));
    //处理导入失败的错误事件
    for (int32_t i = 0; i < parser->getNbErrors(); ++i) {
        logger->log(Level::kERROR, parser->getError(i)->desc());
        return;
    }

    //如果没有报错，就构建一个Engine:
    engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
}
ModelImporter::~ModelImporter(){
}


bool ModelImporter::saveToFile(const std::string & path_to_save){
    size_t pos = path_to_save.find_last_of('.');
    if(path_to_save.substr(pos, path_to_save.length() - pos) != ".engine"){
        logger->log(Level::kERROR, "Output file is not .engine file");
        return false;
    }
    if(!engine) return false;
    std::shared_ptr<nvinfer1::IHostMemory> memory = std::shared_ptr<nvinfer1::IHostMemory>(engine->serialize());
    std::ofstream p(path_to_save, std::ios::binary); 
    if (!p) { 
        logger->log(Level::kERROR, "could not open output file to save model"); 
        return false; 
    } 
    p.write(reinterpret_cast<const char*>(memory->data()), memory->size()); 
    return true;
}

std::optional<std::shared_ptr<nvinfer1::ICudaEngine>> ModelImporter::Engine(){
    if(!engine){
        return std::nullopt;
    } else return engine;
}
}