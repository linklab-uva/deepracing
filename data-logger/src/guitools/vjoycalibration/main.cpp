#include "vjoycalibrationwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    VjoyCalibrationWindow w;
    w.show();

    return a.exec();
}
