#include "inferencer.h"
#include "tensor.h"

int main(int argc, char ** argv){
    if(argc < 2) return 1;
    std::string path = argv[1];
    std::string model_name = path.substr(0, path.find_last_of('.'));
    //std::cout << model_name << std::endl;
    
    try{
        GcRT::Inferencer syci(path);
        std::cout << "build done" << std::endl;
        float arr[3 * 224 * 224];
        for(int i = 0;i < 3 * 224 * 224;i++){
            arr[i] = rand() / (float)RAND_MAX;
        }

        auto t1 = GcRT::createTensorByType(nvinfer1::DataType::kFLOAT, {4, {1, 3, 224, 224}});
        auto ret = syci.inference(std::move(t1));
        if(ret.has_value()){
            std::cout << ret.value()->size() << std::endl;
            std::cout << (int)ret.value()->type() << std::endl;
            std::cout << ret.value()->dims().nbDims << std::endl;
            std::cout << ret.value()->dims().d[0] << std::endl;
            std::cout << ret.value()->dims().d[1] << std::endl;
            std::cout << ret.value()->dims().d[2] << std::endl;
        } else {
            std::cout << "inference failed" << std::endl;
        }

    } catch(const std::exception & e){
        std::cout << e.what() << std::endl;
    }


    
    return 0;
}