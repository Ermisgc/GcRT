#pragma once
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <string_view>
#include <map>
#include "log_base.h"
namespace gcspdlog
{
    class Formatter
    { 
    private:
        //store the function and its parameter.
        std::vector<std::pair<char, std::string>> pattern;

        //temporarily store the message as a global parameter.
        LogMsg::ptr temp_msg;
    public:
        using ptr = std::shared_ptr<Formatter>;
        Formatter();
        Formatter(const std::string & fmt);
        ~Formatter();

        void setPattern(const std::string & fmt);
        std::string format(LogMsg::ptr msg);
    private:
        std::string getYear(const std::string & _="");
        std::string getMonth(const std::string & _="");
        std::string getDay(const std::string & _="");
        std::string getMessage(const std::string & _="");
        std::string getHour(const std::string & _="");
        std::string getMinute(const std::string & _="");
        std::string getSecond(const std::string & _="");
        std::string getLevel(const std::string & _="");
        std::string getSourceFile(const std::string & _="");
        std::string getFunction(const std::string & _="");
        std::string getLine(const std::string & _="");
        std::string getThreadid(const std::string & _="");
        std::string getString(const std::string & _);
    };
} // namespace gcspdlog
