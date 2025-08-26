#pragma once
#include "logger.h"
namespace gcspdlog
{
    class Registry{
    private:
        Registry();
        ~Registry();
        static std::vector<Logger::ptr> m_loggers;
        static Registry * instance;

    public:
        static Registry* getInstance();
        static Logger::ptr getLoggerByName(const std::string & name);
    }
} // namespace gcspdlog
