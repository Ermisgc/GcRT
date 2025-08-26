#include "network/json.hpp"
#include "inference_mata.h"
#include <fstream>
#include <iostream>

#define print(XX) std::cout << #XX  << ": " << XX << std::endl;

void handle_inference(const std::string & file_name){
    try{
        //解析json
        std::ifstream body(file_name); 
        
        nlohmann::json json = nlohmann::json::parse(body);

        GcRT::InferenceReq req;
        json.get_to(req);

    } catch (const std::exception & e){
        std::cerr << "Catch a exception in handle_infernce: " << e.what() << std::endl;
    }
}


int main(int argc, char ** argv){
    handle_inference(argv[1]);
    return 0;
}