#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <opencv2/videoio.hpp>
#include <chrono> 
using namespace std::chrono;
std::vector<cv::Mat> getImages(const std::string& dir, unsigned int first_image, unsigned int num_images)
{
    std::vector<cv::Mat> rtn;
    unsigned int end = first_image + num_images;
    for(int i = first_image; i < end; i ++)
    {
        std::string file = dir + "/" + "raw_image_" + std::to_string(i) + ".jpg";
        rtn.push_back(cv::imread(file));
    }
    return rtn;
}
int main(int argc, char** argv)
{
    std::cout<<"Hello World!"<<std::endl;
    auto start = high_resolution_clock::now(); 
    std::vector<cv::Mat> images = getImages("/home/ttw2xk/deepf1data/australia_fullview_run2/raw_images", 7, 10);
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<milliseconds>(stop - start); 
    std::cout <<"Took " << duration.count() << " milliseconds"<<std::endl;
    cv::namedWindow("image",cv::WINDOW_AUTOSIZE);
    for(const cv::Mat& mat : images)
    {
        cv::imshow("image", mat);
        cv::waitKey(0);
    }
}