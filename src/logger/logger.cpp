#include "logger/logger.h"
namespace gcspdlog{

Logger::Logger(const std::string & name, Level level, Sink::ptr default_sink)\
    :m_name(name), m_level(level){
        m_sinks.push_back(default_sink);
    }

void Logger::addSink(Sink::ptr sink){
    m_sinks.push_back(sink);
}

void Logger::removeSink(Sink::ptr sink){
    //how to remove a sink?
    for(auto m_sink = m_sinks.begin(); m_sink != m_sinks.end(); m_sink = std::next(m_sink)){
        if(*m_sink == sink) {
            m_sinks.erase(m_sink);
            break;
        }
    }
}

void Logger::removeAllSinks(){
    m_sinks.clear();
}

void Logger::log(LogMsg::ptr message){
    //if the level is small than the logger_level, the logger will not handle it.
    if(message->level > m_level) return;
    for(auto & sink:m_sinks){
        sink->log(message);
    }
}

Logger::~Logger(){

}

AsyncLogger::AsyncLogger(const std::string & name, Level level, Sink::ptr default_sink): Logger(name, level, default_sink), \
   _log_thread(&AsyncLogger::log_worker, this){

}

AsyncLogger::~AsyncLogger(){
    _destroying = true;
    if(_log_thread.joinable()) _log_thread.join();
}

void AsyncLogger::log(LogMsg::ptr msg){
    if(msg->level < m_level) return;
    _lock_free_queue.enqueue(msg);
    _inqueue_logs++;
    _wait_cv.notify_one(); //通知log_worker干活
}

void AsyncLogger::SyncLog(LogMsg::ptr msg){
    if(msg->level < m_level) return;
    for(auto & sink : m_sinks){
        sink->log(msg);
    }
}

void AsyncLogger::log_worker(){
    while(!_destroying){
        {
            //这里只会是一个MPSC，这里加一个设置，让线程等待，不会空转
            std::unique_lock<std::mutex> locker(_wait_mutex);
            _wait_cv.wait_for(locker, std::chrono::microseconds(100), [&]() -> bool {return _destroying || _inqueue_logs.load() > 0;});
        }

        while(_inqueue_logs.load() > 0){
            LogMsg::ptr msg;
            if(!_lock_free_queue.try_dequeue(msg)) break;
            
            for(auto & sink : m_sinks){
                sink->log(msg);
            }

            _inqueue_logs--;
        }
    }
}
}