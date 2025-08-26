#pragma once
#include <string>
#include "./network/json.hpp"
#include <vector>
#include <sstream>
#include <iostream>

namespace GcRT{    
    using json = nlohmann::json;

    struct ModelConfig{
        std::string name;
        std::string version;
    };

    struct InferenceParam{
        int priority;
        int timeout;
        int max_batch_size; 
        std::string output_format;
    };

    struct InputData{
        std::string type;
        std::string uri;
    };

    struct Input{
        std::string name;
        std::string datatype;
        std::string data;
        std::vector<int> shape;
        InputData data_ref;
    };

    struct Output{
        std::string name;
        bool ret;
        std::string postprocessing;
    };

    struct Request{
        std::string _request_id;
        ModelConfig _model_config;
        InferenceParam _infer_param;
        std::vector<Input> _inputs;
        std::vector<Output> _outputs;
    };

    //执行上下文与其元信息
    struct ExecutionContextMeta{
        int batch_size;
        int nb_input;
        int nb_output;
        nvinfer1::IExecutionContext * ctx;
        std::vector<void *> input_ptrs;
        std::vector<void *> output_ptrs;
        
        std::vector<int> input_sizes;
        std::vector<int> output_sizes;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names; 
    };

    //一个流水线任务的元数据
    typedef void (*Callback)(cudaStream_t, cudaError_t, void *);

    //提取出的最核心的推理请求结构体
    struct InferenceReq{
        //输入和输出host指针
        std::vector<void *> h_input_buffer;
        std::vector<size_t> input_sizes;

        std::vector<void *> h_output_buffer;
        std::vector<size_t> output_sizes;

        //执行任务的回调函数
        Callback call_back;

        //任务的优先级
        int priority;

        //会话的回调接口
        void * user_data;
    };  

    void from_json(const json& j, ModelConfig & config){
        j.at("name").get_to(config.name);
        j.at("version").get_to(config.version);
    }

    void from_json(const json& j, InferenceParam & param){
        j.at("priority").get_to(param.priority);
        j.at("timeout").get_to(param.timeout);
        j.at("max_batch_size").get_to(param.max_batch_size);
        j.at("output_format").get_to(param.output_format);
    }

    void from_json(const json& j, InputData & data){
        j.at("type").get_to(data.type);
        j.at("uri").get_to(data.uri);
    }

    void from_json(const json& j, Input & input){
        j.at("name").get_to(input.name);
        j.at("datatype").get_to(input.datatype);
        j.at("data").get_to(input.data);
        j.at("shape").get_to(input.shape);
        j.at("data_ref").get_to(input.data_ref);
    }

    void from_json(const json& j, Output & output){
        j.at("name").get_to(output.name);
        j.at("return").get_to(output.ret);
        j.at("postprocessing").get_to(output.postprocessing);
    }

    void from_json(const json& j, Request& req) {
        j.at("request_id").get_to(req._request_id);
        j.at("model").get_to(req._model_config);  // 修正这里
        j.at("parameters").get_to(req._infer_param);
        j.at("inputs").get_to(req._inputs);
        j.at("outputs").get_to(req._outputs);
    }

    static constexpr char base64_charset_table[64] = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
        'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
        'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
    };

    //TODO:这里应该加上无效字符检测，但是暂时没做
    unsigned char base64_reverse(char input){
        if(input == '+') return 62;
        if(input == '/') return 63;
        if(input >= 'a') return 26 + input - 'a';
        else if(input >= 'A') return input - 'A';
        else return 52 + input - '0';
    }

    std::string base64Encoding(const std::vector<uint8_t> & input){
        std::string encoded;
        int n = input.size();
        encoded.reserve( (4 * n + 2) / 3);  //首先预留一部分空间

        //11111111，11111111，11111111 -> 00111111,00111111,00111111,00111111

        for(int i = 0;i < n; ++i){
            if(i % 3 != 2) continue;
            //else 的情况，可以进行encoding了，把input每三个一组变成base64
            encoded.push_back(base64_charset_table[(input[i - 2] & 0xfc) >> 2]);
            encoded.push_back(base64_charset_table[((input[i - 2] & 0x03) << 4) | ((input[i - 1] & 0xf0) >> 4)]);
            encoded.push_back(base64_charset_table[((input[i - 1] & 0x0f) << 2) | ((input[i] & 0xc0) >> 6)]);
            encoded.push_back(base64_charset_table[input[i] & 0x3f]);
        }

        //最后处理一下，可能会有剩余的字符
        if(n % 3 == 1){  //多了一个字符
            encoded.push_back(base64_charset_table[(input[n - 1] & 0xfc) >> 2]);
            encoded.push_back(base64_charset_table[(input[n - 1] & 0x03) << 4]);
            encoded.push_back('=');
            encoded.push_back('=');
        }else if(n % 3 == 2){  //多了两个字节
            encoded.push_back(base64_charset_table[(input[n - 2] & 0xfc) >> 2]);
            encoded.push_back(base64_charset_table[((input[n - 2] & 0x03) << 4) | ((input[n - 1] & 0xf0) >> 4)]);
            encoded.push_back(base64_charset_table[(input[n - 1] & 0x0f) << 2]);
            encoded.push_back('=');
        }

        return encoded;
    }

    std::vector<uint8_t> base64Decoding(const std::string & input){
        std::vector<uint8_t> decoded;
        int n = input.length();
        decoded.reserve( (3 * n + 3) / 4);  //首先预留一部分空间

        int count = 0;
        //00111111,00111111,00111111,00111111 -> 11111111,11111111,11111111
        //11111111  -> 00111111, 00000011
        //每个字符都是前一个的后6位，前一个的前2位
        unsigned char char_group[4];
        for(char c : input){
            if(c == '=') break; //到了补充位就可以停止了
            if(c > 'z' || (c > 'Z' && c < 'a') || (c < 'A' && c > '9') || (c < '0' && c != '/' && c != '+')){
                std::cerr << "Unknown char when decoding base64." << std::endl;
            }
            //然后是每四个字符一组
            char_group[count++] = base64_reverse(c);
            if(count % 4 == 0){
                decoded.push_back(char_group[0] << 2 | char_group[1] >> 4);
                decoded.push_back(char_group[1] << 4 | char_group[2] >> 2);
                decoded.push_back(char_group[2] << 6 | char_group[3]);
                count = 0;
            }
        }

        //最后处理一下，可能会有剩余的字符
        if(count % 4 == 1){    //里面只剩一个字符了
            decoded.push_back(char_group[0] << 2);
        }else if(count % 4 == 2){
            decoded.push_back(char_group[0] << 2 | char_group[1] >> 4);
            decoded.push_back(char_group[1] << 4);
        } else if(count % 4 == 3){
            decoded.push_back(char_group[0] << 2 | char_group[1] >> 4);
            decoded.push_back(char_group[1] << 4 | char_group[2] >> 2);
            decoded.push_back(char_group[2] << 6);
        }

        if(decoded.back() == 0){
            decoded.pop_back();
        }

        return decoded;
    }

}