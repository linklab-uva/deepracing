#include <pybind11/pybind11.h>

pybind11::bytes scenario_to_tfexample(pybind11::bytes bytesin) {
    return bytesin;
}

PYBIND11_MODULE(waymo_conversions, m) {
    m.doc() = "Simple waymo conversions"; // optional module docstring

    m.def("scenario_to_tfexample", &scenario_to_tfexample, "Do scenario to tfexample conversion in C++");
}