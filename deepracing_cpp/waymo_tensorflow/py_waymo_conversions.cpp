#include <pybind11/pybind11.h>
#include <waymo_open_dataset/data_conversion/scenario_conversion.h>
#include <google/protobuf/util/json_util.h>

pybind11::bytes scenario_to_tfexample(const pybind11::bytes& scenario_bytes) {
    waymo::open_dataset::Scenario scenario;
    scenario.ParseFromString(scenario_bytes);
    // google::protobuf::util::JsonStringToMessage(scenario_json, &scenario);
    // std::string scenario_json;
    // google::protobuf::util::MessageToJsonString(scenario, &scenario_json);
    // std::cout<<scenario_json<<std::endl;
    waymo::open_dataset::MotionExampleConversionConfig config;
    std::map<std::string,int> counters;
    absl::StatusOr<tensorflow::Example> converted_ptr = waymo::open_dataset::ScenarioToExample(scenario, config, &counters);
    const tensorflow::Example& converted = *converted_ptr;
    std::string out;
    // google::protobuf::util::MessageToJsonString(converted, &out);
    converted.SerializeToString(&out);
    return pybind11::bytes(out);
}

PYBIND11_MODULE(py_waymo_conversions, m) {
    m.doc() = "Simple waymo conversions"; // optional module docstring

    m.def("scenario_to_tfexample", &scenario_to_tfexample, "Do scenario to tfexample conversion in C++");
}