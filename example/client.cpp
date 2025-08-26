#include <iostream>
#include "network/client.h"
/**
 * @file client.cpp
 * client将实现用户请求的异步包装和发送，暂时采用HTTP协议
 * @class Client类包装用户的基本信息，包括用户名id、主机名等
 * @class UesrMessage类实现对信息的包装：例如：设定附加文件类型、用户请求的文本，拟前端对服务器发起的信息
 * @class HTTPRequest类负责将所给请求转化为HTTP请求，它提供方法: std::string buildContent(const Client & c, const char * msg, const char * filename);
 * 此外，HTTPRequest类还可以解析所传入的HTTP请求
 * @class HTTPReponse类，负责解析所给的HTTPResponse和生成Response
 * @class TcpClient类，简单实现，只是负责信息的收发
 */

int main(){
    return 0;

}