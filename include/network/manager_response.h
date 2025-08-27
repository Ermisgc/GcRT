#pragma once
#include "json.hpp"

namespace GcRT{
    enum class ManagementOp;
    using json = nlohmann::json;

    ManagementOp string2ManagementOp(const std::string & input);
    std::string managementOp2String(ManagementOp op);

    struct ManagerResponse{
        ManagementOp op;
        std::string model_id;
        bool success;
        int error_code;
        std::string message;
    };

    void from_json(const json & j, ManagerResponse & response){
        j.at("model_id").get_to(response.model_id);
        j.at("success").get_to(response.success);
        j.at("error_code").get_to(response.error_code);
        j.at("message").get_to(response.message);
        response.op = string2ManagementOp(j.at("operation").get<std::string>());
    }

    void to_json(json & j, const ManagerResponse & response){
        j["model_id"] = response.model_id;
        j["success"] = response.success;
        j["error_code"] = response.error_code;
        j["message"] = response.message;
        j["operation"] = managementOp2String(response.op);
    }
}
