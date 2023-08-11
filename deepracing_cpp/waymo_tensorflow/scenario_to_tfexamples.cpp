#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <waymo_open_dataset/data_conversion/scenario_conversion.h>
#include <google/protobuf/util/json_util.h>

#ifdef USE_BOOST_FILESYSTEM
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace po = boost::program_options;
void exit_with_help(po::options_description& desc)
{
        std::stringstream ss;
        ss << "Convert waymo scenario protos to tf_example protos. Command line arguments are as follows:" << std::endl;
        desc.print(ss);
        std::printf("%s", ss.str().c_str());
        exit(0); // @suppress("Invalid arguments")
}

int main(int argc, char** argv)
{
    std::cout << "Hello World!" << std::endl;
    po::options_description desc("Options");
    std::string inputdir;
    try{
            desc.add_options()
            ("help,h", "Displays options and exits")
            ("inputdir", po::value<std::string>(&inputdir), "Where to load scenario files from")
            ;
        po::variables_map vm;
        boost::program_options::store(boost::program_options::command_line_parser(argc, argv).
                                                            options(desc).
                                                            run(), vm);
        boost::program_options::notify(vm);
        if (vm.find("help") != vm.end()) {
                exit_with_help(desc);
        }
    }catch(boost::exception& e){
            exit_with_help(desc);
    }
    if (inputdir.empty())
    {
        std::cerr<<"ERROR: Must specify input directory with --inputdir <path>"<<std::endl;
        exit(-1);
    }
    
    fs::path rawprotopath = fs::canonical(fs::path(inputdir));
    fs::path trainingpath = rawprotopath/fs::path("training");
    fs::path validationpath = rawprotopath/fs::path("validation");
    fs::path testingpath = rawprotopath/fs::path("testing");
    
    std::cout << trainingpath.string() << std::endl;
    std::cout << validationpath.string() << std::endl;
    std::cout << testingpath.string() << std::endl;
   
    fs::path parentpath =  rawprotopath.parent_path();
    fs::path tfexamplepath = parentpath / fs::path("tf_example");
    fs::path trainingpath_out = tfexamplepath/fs::path("training");
    fs::path validationpath_out = tfexamplepath/fs::path("validation");
    fs::path testingpath_out = tfexamplepath/fs::path("testing");
    std::cout << trainingpath_out.string() << std::endl; 
    std::cout << validationpath_out.string() << std::endl; 
    std::cout << testingpath_out.string() << std::endl; 
    fs::create_directories(trainingpath_out);
    std::vector<fs::path> subpaths;
    for (const fs::directory_entry& dir_entry : 
        fs::directory_iterator(trainingpath))
    {
        if(dir_entry.is_directory())
        {
            subpaths.push_back(dir_entry.path().stem());
        }
    }
    for (const fs::path& subpath : subpaths)
    {
        std::cout<<subpath<<std::endl;
        fs::path recordfilepath = trainingpath_out/fs::path("training." + subpath.string());
        std::fstream recordfile(recordfilepath.string(), std::ios::out | std::ios::binary);
        waymo::open_dataset::MotionExampleConversionConfig config;
        for (const fs::directory_entry& dir_entry : 
            fs::directory_iterator(trainingpath/subpath))
        {
           if(dir_entry.is_regular_file() && dir_entry.path().extension()==".pb")
           {
            waymo::open_dataset::Scenario scenario;
            std::fstream scenariofile(dir_entry.path().string(), std::ios::in | std::ios::binary);
            scenario.ParseFromIstream(&scenariofile);
            scenariofile.close();
            std::map<std::string, int> counters;
            absl::StatusOr<tensorflow::Example> converted_ptr = waymo::open_dataset::ScenarioToExample(scenario, config, &counters);
            const tensorflow::Example& converted = *converted_ptr;
            // std::string converted_json;
            // google::protobuf::util::MessageToJsonString(converted, &converted_json);
            // std::cout<<converted_json<<std::endl;
           }
        }
        recordfile.close();
        
    }


}
