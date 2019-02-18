#include <pybind11/pybind11.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

#include <string>
#include <iostream>

#include "ndarray_converter.h"
#include <pybind11/stl.h>
#include <exception>
namespace py = pybind11;

cv::Mat readImage(const std::string& image_name, int flag)
{
    return cv::imread(image_name, flag);
}

std::vector<cv::Mat> readImages(const std::string& prefix, int first_image, int num_images, int flag, const std::vector<int> size)
{
    std::vector<cv::Mat> rtn;
    switch (flag)
    {
        case cv::ImreadModes::IMREAD_GRAYSCALE:
            rtn = std::vector<cv::Mat>(num_images, cv::Mat(size[0], size[1], CV_8U) );
            break;
        case cv::ImreadModes::IMREAD_COLOR:
            rtn = std::vector<cv::Mat>(num_images, cv::Mat(size[0], size[1], CV_8UC3) );
            break;
        default:
            throw std::runtime_error("Only IMREAD_GRAYSCALE and IMREAD_COLOR are currently supported");
    }
    
    unsigned int end = first_image + num_images;
    for(int i = first_image; i < end; i ++)
    {
        std::string file = prefix + std::to_string(i+1) + ".jpg";
        cv::resize( cv::imread(file, flag), rtn[i-first_image], cv::Size2i(size[1], size[0]) ) ;
    }
    return rtn;
}
std::vector<cv::Mat> readImageFlows(const std::string& prefix, int first_image, int num_images, const std::vector<int> size)
{
    std::vector<cv::Mat> rtn(num_images, cv::Mat(size[0], size[1], CV_32FC2));
    std::string file = prefix + std::to_string(first_image+1) + ".jpg";
    cv::Mat first(size[0], size[1], CV_8U);
    cv::Size2i insize(size[1], size[0]);
    cv::resize(cv::imread(file, cv::ImreadModes::IMREAD_GRAYSCALE), first, insize);
    unsigned int start = first_image + 2;
    unsigned int end = first_image + num_images;
    for(int i = start; i < end; i ++)
    {
        file = prefix + std::to_string(i) + ".jpg";
        cv::Mat second(size[0], size[1], CV_8U);
        cv::resize( cv::imread( file, cv::ImreadModes::IMREAD_GRAYSCALE) , second, insize );
        cv::calcOpticalFlowFarneback( first, second, rtn[i-start], 0.5, 3, 15, 3, 5, 1.2, 0 );
        first = second;
    }
    return rtn;
}
PYBIND11_PLUGIN(deepf1_image_reading) 
{
    NDArrayConverter::init_numpy();
    py::module m("deepf1_image_reading", "pybind11 plugin for some opencv stuff");
    m.def("readImage", &readImage, "Read an image from the provided file path", py::arg("image file"), py::arg("imread flag"));
    m.def("readImages", &readImages, "Read a sequence of images from the provided prefix", py::arg("image_prefix"), py::arg("first_image"), py::arg("num_images"), py::arg("imread_flag") , py::arg("image_size"));
    m.def("readImageFlows", &readImageFlows, "Read a sequence of optical flows", py::arg("image_prefix"), py::arg("first_image"), py::arg("num_images"), py::arg("image_size") );
    
    return m.ptr();
}

