#pragma once
#include <chrono>
#include <memory>
#include <iostream>
#include <string>
#include <thread>

//define some macros easy to use
#define QuickMsg(msg, level) std::make_shared<gcspdlog::LogMsg>(msg, level, gcspdlog::SourceLoc(__FILE__, __func__, __LINE__))
#define QuickMsgOFF(msg) QuickMsg(msg, gcspdlog::GCSPDLOG_LEVEL_OFF)
#define QuickMsgCRI(msg) QuickMsg(msg, gcspdlog::GCSPDLOG_LEVEL_CRITICAL)
#define QuickMsgERR(msg) QuickMsg(msg, gcspdlog::GCSPDLOG_LEVEL_ERROR)
#define QuickMsgWARN(msg) QuickMsg(msg, gcspdlog::GCSPDLOG_LEVEL_WARN)
#define QuickMsgDEBUG(msg) QuickMsg(msg, gcspdlog::GCSPDLOG_LEVEL_DEBUG)
#define QuickMsgINFO(msg) QuickMsg(msg, gcspdlog::GCSPDLOG_LEVEL_INFO)
#define QuickMsgTRACE(msg) QuickMsg(msg, gcspdlog::GCSPDLOG_LEVEL_TRACE)

namespace gcspdlog
{
#ifdef __Linux__
    //a nickname of std::chrono::high_resolution_clock
    using LogClock = std::chrono::high_resolution_clock;
#endif

#ifdef _WIN32
    using LogClock = std::chrono::system_clock;
#endif 

    enum Level{
        GCSPDLOG_LEVEL_TRACE = 0,
        GCSPDLOG_LEVEL_DEBUG = 1,
        GCSPDLOG_LEVEL_INFO = 2,
        GCSPDLOG_LEVEL_WARN = 3,
        GCSPDLOG_LEVEL_ERROR = 4,
        GCSPDLOG_LEVEL_CRITICAL = 5,
        GCSPDLOG_LEVEL_OFF = 6
    };

    struct SourceLoc{
        const char * filename{nullptr};
        const char * funcname{nullptr};
        uint32_t line{0};

        SourceLoc(const char * _filename, const char * _funcname, uint32_t _line): filename(_filename), funcname(_funcname), line(_line){}
        SourceLoc():filename(__FILE__), funcname(__func__), line(__LINE__){}
        //if not valid, the information will not be written
        bool valid(){
            return line > 0 && filename && funcname;
        }
    };

    struct LogMsg{
        std::thread::id thread_id;
        Level level{Level::GCSPDLOG_LEVEL_OFF};
        std::tm* time{nullptr};
        SourceLoc source;
        std::string msg;

        LogMsg() = default;
        LogMsg(const std::string & _msg, Level _level = GCSPDLOG_LEVEL_OFF, SourceLoc _src = SourceLoc(), LogClock::time_point _time = LogClock::now(), std::thread::id _thread = std::this_thread::get_id()):\
            msg(_msg), level(_level), source(_src), thread_id(_thread){
            auto time_t_value = LogClock::to_time_t(_time);
            time = std::localtime(&time_t_value);
        }
        using ptr = std::shared_ptr<LogMsg>;
    };
} // namespace gcspdlog