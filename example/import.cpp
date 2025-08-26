#include "model_importer.h"
#include <string>
#include <iostream>

int main(int argc, char ** argv){
    if(argc < 2) return 1;
    std::string path = argv[1];
    std::string model_name = path.substr(0, path.find_last_of('.'));
    std::cout << model_name << std::endl;
    GcRT::ModelImporter importer(path, 0);
    if(!importer.saveToFile(model_name + ".engine")){
        std::cout << "保存成功" << std::endl;
    }
    return 0;
}