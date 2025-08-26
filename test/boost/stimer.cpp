#define BOOST_ASIO_NO_DEPRECATED
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include "network/json.hpp"
#include <iostream>
#include <vector>
#include <memory>

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

#define BUF_SIZE 1024

class HttpSession : public std::enable_shared_from_this<HttpSession> {
public:
  HttpSession(tcp::socket socket) : socket_(std::move(socket)) {
  }

  void Start() {
    DoRead();
  }

  void DoRead() {
    auto self(shared_from_this());
    socket_.async_read_some(
        boost::asio::buffer(buffer_),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            DoWrite(length);
          }
        }
    );
  }

  void DoWrite(std::size_t length) {
    auto self(shared_from_this());
    boost::asio::async_write(
        socket_,
        boost::asio::buffer(buffer_, length),
        [this, self](boost::system::error_code ec, std::size_t length) {
          if (!ec) {
            DoRead();
          }
        });
  }

private:
  tcp::socket socket_;
  std::array<char, BUF_SIZE> buffer_;
};

class Server{
public:
    Server(asio::io_context & ioc, std::uint16_t port): _acceptor(ioc, tcp::endpoint(tcp::v4(), port)){
        DoAccept();
    }

private:
    void DoAccept(){
        _acceptor.async_accept(
            [this](boost::system::error_code ec, tcp::socket socket){
                if(!ec){
                    std::make_shared<HttpSession>(std::move(socket))->Start();
                }
                DoAccept();
            }

        );
    }

private:
    tcp::acceptor _acceptor;
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <port>" << std::endl;
        return 1;
    }

    unsigned short port = std::atoi(argv[1]);
    boost::asio::io_context ioc;

    // 创建 Acceptor 侦听新的连接
    // tcp::acceptor acceptor(ioc, tcp::endpoint(tcp::v4(), port));
    Server server(ioc, port);
    ioc.run();
    return 0;
}
