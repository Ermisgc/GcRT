#include "logger/sink.h"
namespace gcspdlog{
Sink::Sink():m_level(Level::GCSPDLOG_LEVEL_OFF), m_formatter(std::make_shared<Formatter>()){}

Sink::Sink(Level level, const Formatter::ptr Formatter):m_level(level), m_formatter(Formatter){}

Sink::~Sink(){}

void Sink::log(LogMsg::ptr msg){
    if(msg->level > this->m_level) return;
    std::cout << m_formatter->format(msg) << std::endl;
}

void Sink::setFormatter(const Formatter::ptr formatter){
    if(formatter != nullptr){
        this->m_formatter = formatter;
    } 
}

FileSink::FileSink():file_name("log.txt"), Sink(){}

FileSink::FileSink(const std::string & file, Level Level, const Formatter::ptr formatter):file_name(file), Sink(Level, formatter){

}

FileSink::~FileSink(){
}
        
void FileSink::log(LogMsg::ptr msg) {
    if(msg->level < m_level) return;
    if(!_file_stream){
        _file_stream = std::make_shared<std::ofstream>(file_name, std::ios_base::out | std::ios_base::app);
        if(!_file_stream->is_open()){
            std::cout << "The file:[" << file_name << "] doesn't exist" << std::endl;
            _file_stream.reset();
            return;
        }
    }
    (*_file_stream) << this->m_formatter->format(msg) << std::endl; 
}

}