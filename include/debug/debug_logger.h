#include "logger/logger.h"
#ifndef DEBUG_LOGGER_H
#define DEBUG_LOGGER_H
namespace GcRT{
namespace dbg{
    class DebugLogger :public gcspdlog::AsyncLogger{
        gcspdlog::Formatter::ptr fmt;
    public:
        DebugLogger() : AsyncLogger("debug", gcspdlog::Level::GCSPDLOG_LEVEL_DEBUG){
            removeAllSinks();
            fmt = std::make_shared<gcspdlog::Formatter>("[%l][%y-%m-%d-%h-%i-%e][%f, tid:%t, line:%n]: %s");
            auto file_sink = std::make_shared<gcspdlog::FileSink>("./nvprof_log/debug_log.txt", gcspdlog::GCSPDLOG_LEVEL_DEBUG, fmt);   
            auto sink = std::make_shared<gcspdlog::Sink>(gcspdlog::GCSPDLOG_LEVEL_DEBUG, fmt);  
            addSink(file_sink);
            addSink(sink);
        }
    };

    
    class LockLogger : public gcspdlog::AsyncLogger{
        gcspdlog::Formatter::ptr fmt;
    public:
    };
}

//放到外部使用
using namespace dbg;
}

#ifdef DEBUG
    std::shared_ptr<GcRT::DebugLogger> __debug_logger = std::make_shared<GcRT::DebugLogger>();  //在调试时设置为全局变量
#define DBG_LOG(msg) __debug_logger->SyncLog(QuickMsgDEBUG(msg));
#else
#define DBG_LOG(msg)
#endif

#endif