#pragma once
#include "json.hpp"
#include <NvInfer.h>
#include <fstream>

namespace GcRT{
    using nvinfer1::DataType;
    using nlohmann::json;

    std::string base64Decoding(const std::string & input);
    std::string base64Encoding(const std::string & input);

    DataType string2Datatype(const std::string & input);
    std::string datatype2String(DataType datatype);

    struct Input{
        std::string name;
        std::vector<int> shape;
        DataType datatype;
        std::string data;
    };

    struct Return{
        std::string name;
        std::string postprocessing;
    };

    struct Parameters{
        int priority;
        int timeout;
    };

    struct Request{
        std::string model_id;
        uint32_t request_id;
        std::vector<Input> inputs;
        std::vector<Return> returns;
    };


    void from_json(const json& j, Input & input){
        j.at("name").get_to(input.name);
        j.at("shape").get_to(input.shape);

        std::string data_type;
        j.at("datatype").get_to(data_type);
        input.datatype = string2Datatype(data_type);
        try{
            j.at("data").get_to(input.data);
            input.data = base64Decoding(input.data);
        } catch(const std::out_of_range & e) {
            std::string uri;
            j.at("data").at("uri").get_to(uri);
            input.data = readUriToData(uri);
        }
    }

    void from_json(const json & j, Return & ret){
        j.at("name").get_to(ret.name);
        j.at("postprocessing").get_to(ret.postprocessing);
    }

    void from_json(const json & j, Parameters & param){
        j.at("priority").get_to(param.priority);
        j.at("timeout").get_to(param.timeout);
    }

    void from_json(const json & j, Request & req){
        j.at("model_id").get_to(req.model_id);
        j.at("request_id").get_to(req.request_id);
        j.at("inputs").get_to(req.inputs);
        j.at("returns").get_to(req.returns);
    }

    void to_json(json & j, const Input & input){
        j["name"] = input.name;
        j["shape"] = input.shape;
        j["datatype"] = datatype2String(input.datatype);
        j["data"] = base64Encoding(input.data);
    }

    void to_json(json & j, const Return & ret){
        j["name"] = ret.name;
        j["postprocessing"] = ret.postprocessing;
    }

    void to_json(json & j, const Parameters & param){
        j["priority"] = param.priority;
        j["timeout"] = param.timeout;
    }

    void to_json(json & j, const Request & req){
        j["model_id"] = req.model_id;
        j["request_id"] = req.request_id;
        j["inputs"] = req.inputs;
        j["returns"] = req.returns;
    }

    std::string readUriToData(const std::string & uri){
        std::ifstream file(uri);
        std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        return data;
    }
}