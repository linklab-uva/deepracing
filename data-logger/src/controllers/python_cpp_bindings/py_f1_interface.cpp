#include "f1_datalogger/controllers/f1_interface_factory.h"
#include <pybind11/pybind11.h>
class PyF1Interface {
    public:
    PyF1Interface(unsigned int device_id=1)
    {
        interface = deepf1::F1InterfaceFactory::getDefaultInterface(device_id);
    }
    void pushDRS()
    {
        interface->pushDRS();
    }
    void setControl(float steering, float throttle, float brake)
    {
        current_control = deepf1::F1ControlCommand(steering, throttle, brake);
        interface->setCommands(current_control);
    }
    private:
        deepf1::F1ControlCommand current_control;
        std::shared_ptr<deepf1::F1Interface> interface;

};
namespace py = pybind11;

PYBIND11_MODULE(py_f1_interface, m)
{
    py::class_<PyF1Interface> py_f1_interface(m, "F1Interface");
    py_f1_interface.def(py::init<const unsigned int>())
    .def("pushDRS", &PyF1Interface::pushDRS)
    .def("setControl", &PyF1Interface::setControl);
}