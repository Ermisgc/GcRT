#pragma once
#include "formatter.h"
#include <fstream>

namespace gcspdlog{
    // Sink corresponds to Appender in Log4J, which depends the logging method in different destination.
    // this could be used directly(as a stdoutSink), or used by deriving a new class for custom.
    class Sink{
    protected:
        Formatter::ptr m_formatter;
        Level m_level;
    public:
        using ptr = std::shared_ptr<Sink>;
        Sink();
        Sink(Level level, const Formatter::ptr formatter = std::make_shared<Formatter>());
        void setFormatter(const Formatter::ptr formatter);
        virtual ~Sink();

        virtual void log(LogMsg::ptr msg);
    };

    class FileSink: public Sink{
        std::string file_name;
        std::shared_ptr<std::ofstream> _file_stream; 
    public:
        using ptr = std::shared_ptr<FileSink>;
        FileSink();
        FileSink(const std::string & file, Level Level = Level::GCSPDLOG_LEVEL_OFF, const Formatter::ptr formatter = std::make_shared<Formatter>());
        ~FileSink();

        void log(LogMsg::ptr msg) override;
    };
}