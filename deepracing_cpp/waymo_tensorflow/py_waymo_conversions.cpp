#include <pybind11/pybind11.h>
#include <waymo_open_dataset/data_conversion/scenario_conversion.h>
#include <google/protobuf/util/json_util.h>

pybind11::bytes scenario_to_tfexample(const pybind11::bytes& scenario_bytes, const pybind11::bytes& conversion_config_bytes) {
    waymo::open_dataset::Scenario scenario;
    scenario.ParseFromString(scenario_bytes);
    waymo::open_dataset::MotionExampleConversionConfig config;
    config.ParseFromString(conversion_config_bytes);
    std::map<std::string,int> counters;
    absl::StatusOr<tensorflow::Example> converted_ptr = waymo::open_dataset::ScenarioToExample(scenario, config, &counters);
    std::string converted_str;
    converted_ptr->SerializeToString(&converted_str);
    return pybind11::bytes(converted_str);
}

PYBIND11_MODULE(py_waymo_conversions, m) {
    m.doc() = "Simple waymo conversions"; // optional module docstring

    m.def("scenario_to_tfexample", &scenario_to_tfexample, "Do scenario to tfexample conversion in C++");
}