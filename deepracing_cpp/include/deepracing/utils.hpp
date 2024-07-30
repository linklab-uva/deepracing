#ifndef DEEPRACING_CPP__UTILS_H_
#define DEEPRACING_CPP__UTILS_H_
#include <map>
#include <string>

namespace deepracing
{
    class Utils
    {
        public:
            static std::map<std::int8_t, std::string> trackNames();
    };
    // inline std::map<std::int8_t, std::string> trackNames();
}



#endif