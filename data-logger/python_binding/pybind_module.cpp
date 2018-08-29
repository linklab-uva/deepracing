#include <ndarray_converter.h>
#include <screen_video_capture.h>
namespace deepf1_pybind{
	class ScreenVideoCapturePybind {
	public:
		ScreenVideoCapturePybind() {
		
		}
		cv::Mat read() {
			return svc_imp_.read();
		}
		void open(std::string application, int x, int y, int width, int height) {
			cv::Rect2d capture_area(x, y, width, height);
			svc_imp_.open(application, capture_area);
		}
	protected:
		deepf1::screen_video_capture svc_imp_;
	};
}
PYBIND11_MODULE(pyf1_datalogger, m) {
	NDArrayConverter::init_numpy();
	m.doc() = R"pbdoc(

        Pybind11 plugin for the f1 datalogger

        -----------------------

    )pbdoc";
	pybind11::class_<deepf1_pybind::ScreenVideoCapturePybind> ScreenVideoCapture(m, "ScreenVideoCapture");
	ScreenVideoCapture.
		def(pybind11::init<>()).
		def("open",&deepf1_pybind::ScreenVideoCapturePybind::open).
		def("read", &deepf1_pybind::ScreenVideoCapturePybind::read, "Screencaps the application");

	#ifdef VERSION_INFO
		m.attr("__version__") = VERSION_INFO;
	#else
		m.attr("__version__") = "dev";
	#endif
}
