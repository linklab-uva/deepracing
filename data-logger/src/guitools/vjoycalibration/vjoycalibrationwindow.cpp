#include "vjoycalibrationwindow.h"
#include "ui_vjoycalibrationwindow.h"

VjoyCalibrationWindow::VjoyCalibrationWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::VjoyCalibrationWindow)
{
    ui->setupUi(this);
}

VjoyCalibrationWindow::~VjoyCalibrationWindow()
{
    delete ui;
}
