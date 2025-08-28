#pragma once
#include <NvInfer.h>
#include <string>
#include "json.hpp"

namespace GcRT{
    using nlohmann::json;
    using nvinfer1::DataType;

    std::string datatype2String(nvinfer1::DataType type);

    std::string base64Encoding(const std::string & input);

    struct Output{
        std::string name;
        std::vector<int> shape;
        std::string data;
        nvinfer1::DataType type;
    };

    struct ResponseParameters{
        int inference_time;
    };
    
    struct Response {
        std::string model_id;
        std::string request_id;
        ResponseParameters parameters;
        std::vector<Output> outputs;
        int error_code;
        std::string error_message;
    };

    void from_json(const json & j, Response & response){
        j.at("model_id").get_to(response.model_id);
        j.at("request_id").get_to(response.request_id);
        j.at("parameters").get_to(response.parameters);
        j.at("outputs").get_to(response.outputs);
        j.at("error_code").get_to(response.error_code);
        j.at("error_message").get_to(response.error_message);
    }

    void from_json(const json & j, Output & output){
        j.at("name").get_to(output.name);
        j.at("shape").get_to(output.shape);
        j.at("data").get_to(output.data);
        j.at("datatype").get_to(output.type);
    }

    void from_json(const json & j, ResponseParameters & param){
        j.at("inference_time").get_to(param.inference_time);
    }

    void to_json(json & j, const Response & response){
        j["model_id"] = response.model_id;
        j["request_id"] = response.request_id;
        j["parameters"] = response.parameters;
        j["outputs"] = response.outputs;
        j["error_code"] = response.error_code;
        j["error_message"] = response.error_message;
    }

    void to_json(json & j, const ResponseParameters & param){
        j["inference_time"] = param.inference_time;
    }

    void to_json(json & j, const Output & output){
        j["name"] = output.name;
        j["shape"] = output.shape;
        j["data"] = base64Encoding(output.data);
        j["datatype"] = datatype2String(output.type);
    }
}