#include "logger/formatter.h"
#include <limits.h>
namespace gcspdlog{
Formatter::Formatter(){
    //[日志等级][年-月-日-时-分-秒][源文件-行]:信息
    this->setPattern("[%l][%y-%m-%d-%h-%i-%e][%o,line:%n]: %s");
}

Formatter::Formatter(const std::string & fmt){
    this->setPattern(fmt);
}
Formatter::~Formatter(){

}

void Formatter::setPattern(const std::string & fmt){
    this->pattern.clear();
    // int last
    int left = 0, right = 0, n = fmt.size();
    for(int i = 0;i < n - 1; ++i){
        if(fmt[i] != '%') continue;
        else{
            //push_back the current string
            right = i;
            if(right > left){
                this->pattern.emplace_back(std::make_pair('.', fmt.substr(left, right - left)));
            }

            switch (fmt[i + 1])
            {
            case 's':case 'y':case 'm':case 'd':case 'h':case 'i':case 'e':case 'l':\
            case 'o':case 'f':case 'n':case 't': 
                this->pattern.emplace_back(std::make_pair(fmt[i + 1], ""));
            default:
                break;
            }

            i++;
            left = i + 1;
        }
    }
    if(n > left) this->pattern.emplace_back(std::make_pair('.', fmt.substr(left, n - left)));
}

std::string Formatter::format(LogMsg::ptr lmsg){
    this->temp_msg = lmsg;

    //use chars mapping to functions, the functions include formatter function and a raw transmitting function.
    static std::map<char, std::function<std::string(const std::string &)>> fmt_map = {
#define XX(func_char, func) {func_char, [this](const std::string & s){return this->func(s);}}
    XX('s', getMessage),
    XX('y', getYear),
    XX('m', getMonth),
    XX('d', getDay),
    XX('h', getHour),
    XX('i', getMinute),
    XX('e', getSecond),
    XX('l', getLevel),
    XX('o', getSourceFile),
    XX('f', getFunction),
    XX('n', getLine),
    XX('t', getThreadid),
    XX('.', getString)
#undef XX
    };
    std::string ans;
    for(auto & pr: pattern){
        ans += fmt_map.at(pr.first)(pr.second);
    }
    return ans;
}

std::string Formatter::getYear(const std::string &){
    return std::to_string(temp_msg->time->tm_year + 1900);
}

std::string Formatter::getMonth(const std::string &){
    return std::to_string(temp_msg->time->tm_mon + 1);
}
std::string Formatter::getDay(const std::string &){
    return std::to_string(temp_msg->time->tm_mday);
}
std::string Formatter::getMessage(const std::string &){
    return temp_msg->msg;
}
std::string Formatter::getHour(const std::string &){
    return std::to_string(temp_msg->time->tm_hour);
}
std::string Formatter::getMinute(const std::string &){
    return std::to_string(temp_msg->time->tm_min);
}
std::string Formatter::getSecond(const std::string &){
    return std::to_string(temp_msg->time->tm_sec);
}
std::string Formatter::getLevel(const std::string &){
    static std::map<Level, std::string_view> level_map = {
        {Level::GCSPDLOG_LEVEL_TRACE,  std::string_view("TRACE", 5)},
        {Level::GCSPDLOG_LEVEL_DEBUG,  std::string_view("DEBUG", 5)},
        {Level::GCSPDLOG_LEVEL_INFO, std::string_view("INFO", 4)},
        {Level::GCSPDLOG_LEVEL_WARN, std::string_view("WARN", 4)},
        {Level::GCSPDLOG_LEVEL_ERROR, std::string_view("ERROR", 5)},
        {Level::GCSPDLOG_LEVEL_CRITICAL, std::string_view("CRITICAL", 8)},
        {Level::GCSPDLOG_LEVEL_OFF, std::string_view("OFF", 3)}
    };
    return std::string(level_map.at(temp_msg->level));
}

std::string Formatter::getSourceFile(const std::string &){
    return temp_msg->source.valid() ? temp_msg->source.filename : "";
}
std::string Formatter::getFunction(const std::string &){
    return temp_msg->source.valid() ? temp_msg->source.funcname : "";
}
std::string Formatter::getLine(const std::string &){
    return temp_msg->source.valid() ? std::to_string(temp_msg->source.line) : "";
}
std::string Formatter::getThreadid(const std::string &){
    return std::to_string(*reinterpret_cast<unsigned int*>(&temp_msg->thread_id));
}

std::string Formatter::getString(const std::string & str){
    return str;
}

}