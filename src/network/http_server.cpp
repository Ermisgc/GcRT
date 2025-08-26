#include "network/http_server.h"

namespace GcRT{
    HttpServer::HttpServer(unsigned short port): _acceptor(_main_ioc, {tcp::v4(), port}), _next_to_index(0){
        std::size_t _num_threads = std::thread::hardware_concurrency();
        for(size_t i = 0;i < _num_threads - 1; ++i){
            _worker_iocs.emplace_back(std::make_unique<asio::io_context>());   //创建ioc
            auto wg = asio::make_work_guard(_worker_iocs[i]);
            _work_guards.emplace_back(std::move(wg));
        }
    }

    HttpServer::~HttpServer(){
        stop();
        for(auto & t: _worker_threads){
            if(t.joinable()) t.join();
        }    
    }

    void HttpServer::run(){
        for(size_t i = 0;i < _num_threads - 1; ++i){
            _worker_threads.emplace_back([this, &i](){
                _worker_iocs[i]->run();
            });
        }

        _main_ioc.run();
    }

    void HttpServer::accept(){
        size_t index  = _next_to_index.fetch_add(1) % (_num_threads - 1);
        auto & ioc = _worker_iocs[index];

        auto socket = std::make_shared<tcp::socket>(ioc);

        _acceptor.async_accept(
            *socket,
            [this, &ioc, &socket](beast::error_code ec){
                if(!ec) {
                    std::make_shared<HttpSession>(std::move(socket)) -> run();
                }
                if(_acceptor.is_open()) accept();
            }
        );
    }

    void HttpServer::stop(){
        _main_ioc.stop();
        for(auto & ctx: _worker_iocs){
            ctx->stop();
        }

        _work_guards.clear();
    }
}