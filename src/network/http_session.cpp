#include "network/http_session.h"
#include <boost/asio/dispatch.hpp>
#include "network/json.hpp"
#include <iostream>
#include "inference_meta.h"
#define MAX_BUFFER_SIZE 1024

namespace GcRT{
    using json = nlohmann::json;

    HttpSession::HttpSession(tcp::socket && socket):_stream(std::move(socket)){

    }

    void HttpSession::run(){
        asio::dispatch(_stream.get_executor(), beast::bind_front_handler(&HttpSession::do_read, shared_from_this()));
    }

    void HttpSession::do_read(){
        _req.clear();
        _stream.expires_after(std::chrono::seconds(30));

        //将用户请求异步读取到_req
        http::async_read(_stream, _buffer, _req, beast::bind_front_handler(&HttpSession::on_read, shared_from_this()));
    }

    void HttpSession::on_read(beast::error_code ec, std::size_t bytes_transferred){
        //此时已经读取完成_req，下一步需要解析这个_req
        boost::ignore_unused(bytes_transferred);
        if(ec == http::error::end_of_stream){
            do_close();
        } else if (ec){
            fail(ec, "read");
        } else {
            //处理解析，然后发送
            process_request();
            write_response();
        }
    }

    void HttpSession::process_request(){
        if(_req.method() != http::verb::post){  //并非是一个post请求
            _res = std::make_unique<http::response<http::string_body>>(http::status::method_not_allowed, _req.version());
            _res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
            _res->set(http::field::content_type, "text/json");
            _res->keep_alive(_req.keep_alive());
            _res->body() = "Only POST method is allowed for common user";
            _res->prepare_payload();
            return;
        }

        if(_req.target() != "/infer") {    //URL不是以infer开头
            _res = std::make_unique<http::response<http::string_body>>(http::status::not_found, _req.version());
            _res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
            _res->set(http::field::content_type, "text/json");
            _res->keep_alive(_req.keep_alive());
            _res->body() = "The target is an invalid path";
            _res->prepare_payload();
            return;
        }

        handle_inference();
    }

    void HttpSession::write_response(){
        bool keep_alive = _res->keep_alive();
        http::message_generator msg(std::move(*_res.get()));
        // Write the response
        beast::async_write(
            _stream,
            std::move(msg),
            beast::bind_front_handler(
                &HttpSession::on_write, shared_from_this(), keep_alive));
    }

    void HttpSession::handle_inference(){
        try{
            //解析json
            auto body = _req.body();
            auto json = nlohmann::json::parse(body);
            Request ir;
            json.get_to(ir);

        } catch (const std::exception & e){
            std::cerr << "Catch a exception in handle_infernce: " << e.what() << std::endl;
            _res = std::make_unique<http::response<http::string_body>>(http::status::bad_request, _req.version());
            _res->set(http::field::server, BOOST_BEAST_VERSION_STRING);
            _res->set(http::field::content_type, "text/json");
            _res->keep_alive(_req.keep_alive());
            _res->body() = e.what();
            _res->prepare_payload();
        }
    }

    void HttpSession::fail(beast::error_code ec, const char * what){
        std::cerr << what << ": " << ec.message() << std::endl;
    }

    void HttpSession::do_close(){
        beast::error_code ec;
        _stream.socket().shutdown(tcp::socket::shutdown_send, ec);
    }  

    void HttpSession::on_write(bool keep_alive, beast::error_code ec, std::size_t bytes_transferred){
        boost::ignore_unused(bytes_transferred);

        if(ec) return fail(ec, "write");
        if(! keep_alive) return do_close();

        do_read();
    }
    
}