#include <WinSock2.h>
#include <string>

#ifndef uint32_t
#define uint32_t unsigned int
#endif


namespace GcRT{
    namespace network{
    class Client{
        uint32_t usr_id_ = 0;
        std::string hostname_;
        in_addr * ip_addr_;

    public:
        Client() = delete;
        ~Client();
        Client(uint32_t usr_id);

        std::string id();

        std::string host();

        std::string ip();
    };
}
}