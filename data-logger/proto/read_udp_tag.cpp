#include "F1UDPData.pb.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <boost/program_options.hpp>

namespace po = boost::program_options;
void exit_with_help(po::options_description& desc)
{
    std::stringstream ss;
    ss << "F1 Read UDP Tag. Command line arguments are as follows:" << std::endl;
    desc.print(ss);
    std::printf("%s", ss.str().c_str());
    exit(0); // @suppress("Invalid arguments")     
}
int main(int argc, char** argv)
{
    std::cout<<"Hello World!"<<std::endl;
    std::string file_in;
    po::options_description desc("Allowed Options");
    try{
        desc.add_options()
        ("help,h", "Displays options and exits")
        ("file,f", po::value<std::string>(&file_in)->required(), "File to Read")
        ;
        if(argc<2)
        {
            exit_with_help(desc);
        }
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
        if (vm.find("help") != vm.end()) {
            exit_with_help(desc);
        }
    }catch(boost::exception& e)
    {
        exit_with_help(desc);
    }

    std::ifstream stream_in;
    stream_in.open(file_in.c_str());
    deepf1::protobuf::F1UDPData data_in;
    data_in.ParseFromIstream(&stream_in);
    stream_in.close();
    std::string data_string = data_in.DebugString();
    printf("Data: \n%s",data_string.c_str());
   // printf("Read some data. Brake: %f. Throttle: %f. Steering: %f\n", data_in.brake(), data_in.throttle(), data_in.steering());
}