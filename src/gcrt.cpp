#include "gcrt.h"
#include <boost/asio.hpp>
#include "network/http_server.h"

namespace GcRT{
    namespace asio = boost::asio;

    GcRT::GcRT() {
        _server = std::make_unique<HttpServer>();
        _engine_scheduler = std::make_unique<EngineScheduler>();
    }

    void GcRT::run(){
        _server->run();
    }

    //submit由HttpSession调用
    void GcRT::submit(const InferenceRequest & req){

    }

    bool loadModel(const std::string & model_id, const ModelImportConfig & config){
        _engine_scheduler->loadModel();
    }

    bool unloadModel(const std::string & model_id){
        
    }

    GcRT & GcRT::getInstance(){
        static GcRT instance;
        return instance;
    }
}