#include <pybind11/pybind11.h>
#include <screen_video_capture.h>
namespace py = pybind11;
PYBIND11_MODULE(pyf1_datalogger, m) {
	m.doc() = R"pbdoc(

        Pybind11 plugin for the f1 datalogger

        -----------------------

    )pbdoc";
	py::class_<deepf1::screen_video_capture> ScreenVideoCapture(m, "ScreenVideoCapture");
	ScreenVideoCapture
		.def("read", &deepf1::screen_video_capture::read);
	ScreenVideoCapture
		.def(py::init<>());
}
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
