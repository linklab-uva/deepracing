#include <pybind11/pybind11.h>
#include <waymo_open_dataset/data_conversion/scenario_conversion.h>

pybind11::bytes scenario_to_tfexample(pybind11::bytes bytesin) {
    waymo::open_dataset::Scenario scenario;
    scenario.ParseFromString(bytesin);
    waymo::open_dataset::MotionExampleConversionConfig config;
    std::map<std::string,int> counters;
    absl::StatusOr<tensorflow::Example> converted_ptr = waymo::open_dataset::ScenarioToExample(scenario, config, &counters);
    const tensorflow::Example& converted = *converted_ptr;
    std::string out;
    converted.SerializeToString(&out);
    return pybind11::bytes(out.c_str());
}

PYBIND11_MODULE(py_waymo_conversions, m) {
    m.doc() = "Simple waymo conversions"; // optional module docstring

    m.def("scenario_to_tfexample", &scenario_to_tfexample, "Do scenario to tfexample conversion in C++");
}