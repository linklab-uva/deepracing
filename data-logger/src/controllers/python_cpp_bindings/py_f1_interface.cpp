#include "f1_datalogger/controllers/f1_interface_factory.h"
#include <pybind11/pybind11.h>
class PyF1Interface {
    public:
    PyF1Interface(unsigned int device_id=1)
    {
        interface = deepf1::F1InterfaceFactory::getDefaultInterface(device_id);
    }
    void setControl(float steering, float throttle, float brake)
    {
        deepf1::F1ControlCommand command;
        command.brake=brake;
        command.throttle=throttle;
        command.steering=steering;
        interface->setCommands(command);
    }
    private:
        std::shared_ptr<deepf1::F1Interface> interface;

};
namespace py = pybind11;

PYBIND11_MODULE(py_f1_interface, m) {
    py::class_<PyF1Interface> py_f1_interface(m, "F1Interface");
     py_f1_interface.def(py::init<const unsigned int>())
        .def("setControl", &PyF1Interface::setControl)
        ;
}