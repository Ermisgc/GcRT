#include "http_session.h"
namespace GcRT{
    class HttpServer {
        using work_guard = asio::executor_work_guard<asio::io_context::executor_type>;

        tcp::acceptor _acceptor;
        std::atomic<size_t> _next_to_index;  //轮询的方式实现负载均衡
        asio::io_context _main_ioc;  //主线程的ioc
        std::vector<std::unique_ptr<asio::io_context>> _worker_iocs; //工作线程的ioc
        std::vector<work_guard> _work_guards;   //work guard保证io_context的生命周期，不会没事干的时候退出
        std::vector<std::thread> _worker_threads;  //工作线程  

        size_t _num_threads;
    public:
        HttpServer(unsigned short port = 1104);
        ~HttpServer();
        void run();
        void stop();
    private:
        void accept();
    };
}