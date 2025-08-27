#pragma once
#include <json.hpp>

namespace GcRT{
    enum class ManagementOp;
    using json = nlohmann::json;

    ManagementOp string2ManagementOp(const std::string & input);
    std::string managementOp2String(ManagementOp op);

    struct ManagerParameter{
        int priority;
    };

    struct ManagerRequest{
        std::string model_id;
        std::string model_path;
        ManagementOp op;
        std::vector<int> batch_sizes;
        ManagerParameter parameters;
    };

    void from_json(const json & j, ManagerRequest & req){
        j.at("model_id").get_to(req.model_id);
        j.at("model_path").get_to(req.model_path);
        std::string op;
        j.at("operation").get_to(op);
        req.op = string2ManagementOp(op);

        j.at("batch_sizes").get_to(req.batch_sizes);
        j.at("parameters").get_to(req.parameters);
    }

    void from_json(const json & j, ManagerParameter & param){
        j.at("priority").get_to(param.priority);
    }

    void to_json(json & j, const ManagerParameter & param){
        j["priority"] = param.priority;
    }

    void to_json(json & j, const ManagerRequest & req){
        j["model_id"] = req.model_id;
        j["model_path"] = req.model_path;
        j["operation"] = managementOp2String(req.op);
        j["batch_sizes"] = req.batch_sizes;
        j["parameters"] = req.parameters;
    }
}