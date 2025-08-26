#include "logger/loggers.h"
#include <memory>
#include <NvInfer.h>
#include <optional>
#include <vector>

namespace GcRT{
    class ModelImporter{
        using Level = BuildLogger::Severity;
    public:
        ModelImporter() = delete;
        ModelImporter(const std::string & imported_model, \
                        nvinfer1::NetworkDefinitionCreationFlags flag = 0);
        ~ModelImporter();

        std::optional<std::shared_ptr<nvinfer1::ICudaEngine>> Engine();
        bool saveToFile(const std::string & path_to_save);
    private:
        std::unique_ptr<BuildLogger>logger;
        std::unique_ptr<nvinfer1::IBuilder> builder;
        std::unique_ptr<nvinfer1::INetworkDefinition> network;
        std::shared_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IBuilderConfig> config;
        bool isConfigFreshed = false;
    };
}