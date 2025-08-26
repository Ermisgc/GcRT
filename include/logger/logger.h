#pragma once
#include <NvInfer.h>
#include <string>
#include <list>
#include <mutex>
#include <thread>
#include "sink.h"
#include "data_structure/concurrentqueue.h"

namespace gcspdlog {
    //日志的基类，直接用就是同步日志
    class Logger {
    protected:  //后代可继承
        std::string m_name;
        std::list<Sink::ptr> m_sinks;  //若干个sinks
        Level m_level;
    public:
        using ptr = std::shared_ptr<Logger>;
        Logger() = delete;   // we don't allowed a Logger of no name;
        Logger(const std::string & name, Level level = GCSPDLOG_LEVEL_OFF, Sink::ptr default_sink = std::make_shared<Sink>());
        void addSink(Sink::ptr);
        void removeSink(Sink::ptr);
        void removeAllSinks();
        virtual void log(LogMsg::ptr msg);  //it's overridable
        ~Logger();
    };

    /**
     * 异步日志
     */
    class AsyncLogger: public Logger{
    private:
        using LockFreeLogQueue = moodycamel::ConcurrentQueue<LogMsg::ptr>;

        LockFreeLogQueue _lock_free_queue;

        std::thread _log_thread;
        std::mutex _wait_mutex;  //主要用于阻塞线程函数的调用
        std::condition_variable _wait_cv;
        
        std::atomic<bool> _destroying{false};
        std::atomic<int> _inqueue_logs{0};  //计数器，这个主要是用解决MoodyCamel的无锁队列没法获得大小的问题

    public:
        AsyncLogger() = delete;
        AsyncLogger(const std::string & name, Level level = GCSPDLOG_LEVEL_OFF, Sink::ptr default_sink = std::make_shared<Sink>());
        ~AsyncLogger();

        virtual void log(LogMsg::ptr msg) override;

        //用于执行某些同步日志消息
        virtual void SyncLog(LogMsg::ptr msg);

    private:
        void log_worker();
    };
    
} // namespace gcspdlog
