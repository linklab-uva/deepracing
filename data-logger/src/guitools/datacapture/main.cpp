#include "datacapture.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    datacapture w;
    w.show();

    return a.exec();
}
