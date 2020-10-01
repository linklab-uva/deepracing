#include <QCoreApplication>
#include <QSaveFile>
#include <QApplication>
#include <QPixmap>
#include <QImage>
#include <QWindow>
#include <QWidget>
#include <QScreen>
#include <QList>
#include <QFile>
#include <QRect>
#include <QIODevice>
#include <QWidget>
#include <iostream>
#include <Win32WindowEnumeration.h>
#include <algorithm>
#include <chrono>
#include <thread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QList<QScreen*> screens = QApplication::screens();
    QScreen* primaryscreen = screens.at(0);
    QString filepath("D:\\asdf.jpg");
    std::cout <<"There are " << screens.length() << " screens" <<std::endl;
//    QFile file(filepath);
//    file.open(QIODevice::WriteOnly);

    std::vector<deepf1::winrt_capture::Window> g_windows = deepf1::winrt_capture::EnumerateWindows();
    std::vector<deepf1::winrt_capture::Window> filtered_windows;
    std::string search_string = std::string(argv[1]);
    std::for_each(g_windows.begin(), g_windows.end(),[&filtered_windows,search_string](const deepf1::winrt_capture::Window &window)
    {
        if (window.TitleStr().find(search_string) != std::string::npos)
        {
            filtered_windows.push_back(window);
        }
    }
    );
    QWindow* grabwin = QWindow::fromWinId((WId)filtered_windows.at(0).Hwnd());
    QWidget* grabwidget = QWidget::find((WId)filtered_windows.at(0).Hwnd());
    QRect rect = grabwin->geometry();
  //  cv::Mat mat;
   // imcv.create(rect.width(), (rect.height()-32)/2, CV_8UC3);
    while(true)
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> tick = std::chrono::high_resolution_clock::now();
        QRect rect = grabwin->geometry();
        //QPixmap pixmap = primaryscreen->grabWindow(0, rect.x(), rect.y()+32, rect.width(), (rect.height()-32)/2);
        QPixmap pixmap = primaryscreen->grabWindow(grabwin->winId(), 0, 0, rect.width(), rect.height());
        
        //QPixmap pixmap = grabwidget->grab();
        QImage image = pixmap.toImage();
        cv::Mat mat(image.height(), image.width(), CV_8UC4, (uchar*)image.bits(), image.bytesPerLine());
        //QPixmap pixmap = primaryscreen->grabWindow(grabwin->winId());
        std::chrono::time_point<std::chrono::high_resolution_clock> tock = std::chrono::high_resolution_clock::now();
        std::printf("Grabbed a window at (x,y): (%d,%d) of height (width,height): (%d,%d)\n", rect.x(), rect.y(), pixmap.width(), pixmap.height());
        std::chrono::duration<double, std::milli> deltat = tock-tick;
        std::printf("The capture took %f milliseconds\n", deltat.count());
        cv::imshow("ImageGrab", mat);
        cv::waitKey(10);
    }
    //QPixmap pixmap = grabwidget->grab();;
    

  //  QPixmap pixmap = primaryscreen->grabWindow();
    //return a.exec();
}
