#include "network/client.h"

namespace GcRT::network{
    Client::Client(uint32_t usr_id):usr_id_(usr_id){
        WSAData WSAData;
        char hostname[256];
        if(!WSAStartup(MAKEWORD(2,0), &WSAData)){
            if(!gethostname(hostname, sizeof(hostname))){
                hostname_ = std::move(hostname);
                hostent * host = gethostbyname(hostname);
                if(host != nullptr){
                    ip_addr_ = reinterpret_cast<in_addr*>(*host->h_addr_list);
                }
            }
        }
    }

    inline std::string Client::id(){
        return std::to_string(usr_id_);
    }

    inline std::string Client::host(){
        return hostname_;
    }

    inline std::string Client::ip(){
        return inet_ntoa(*ip_addr_);
    }
}