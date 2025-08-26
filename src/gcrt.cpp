#include "gcrt.h"
#include <boost/asio.hpp>
#include "network/http_server.h"
#include "scheduler.h"
#include "pipeline_manager.h"

namespace GcRT{
    namespace asio = boost::asio;

    GcRT::GcRT() {
        _pipeline_manager = std::make_shared<PipelineManager>();
        _engine_scheduler = std::make_unique<Scheduler>(_pipeline_manager);
        _server = std::make_unique<HttpServer>();
    }

    void GcRT::run(){
        _server->run();
    }

    void GcRT::addEngine(const std::string & engine_id, const std::string & engine_path, const std::vector<int> & batch_sizes){
        _engine_scheduler->addEngine(engine_id, engine_path, batch_sizes);
    }

    GcRT & GcRT::getInstance(){
        static GcRT instance;
        return instance;
    }
}