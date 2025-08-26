/**
 * HttpSession代表一个HTTP会话，也可以说是一个HTTP连接，它由一个accept到的socket建立
 * HttpSession负责了消息的read、process、write、close.
 */
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <vector>
namespace GcRT{
    namespace asio = boost::asio;
    namespace beast = boost::beast;
    namespace http = beast::http;
    using tcp = asio::ip::tcp;

    //单个http会话，包括读写、推理器的索引、socket、buffer
    class HttpSession : public std::enable_shared_from_this<HttpSession> { 
        beast::tcp_stream _stream;
        beast::flat_buffer _buffer;
        // std::shared_ptr<std::string const> doc_root_;
        http::request<http::string_body> _req;
        std::unique_ptr<http::response<http::string_body>> _res;


    public:
        explicit HttpSession(tcp::socket && socket);

        //开始异步操作，一个Session内的所有异步操作都需要在一个strand上执行
        //run函数要开始异步执行read_request()
        void run();

    private:

        //每次读取时先清空_request
        //给_stream设置读超时
        //然后执行异步on_read()
        void do_read();

        //根据ec的值判断状态，如果状态ok，执行handle_request()，否则执行相应操作
        void on_read(beast::error_code ec, std::size_t bytes_transferred);

        void process_request();

        //先判断是否要keep_alive，然后
        void write_response();

        void handle_inference();

        void fail(beast::error_code ec, const char * what);

        void do_close();  

        void on_write(bool keep_alive, beast::error_code ec, std::size_t bytes_transferred);
    };
}
