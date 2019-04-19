#include "vjoycalibrationwindow.h"
#include "ui_vjoycalibrationwindow.h"
#include <QMessageBox>
#include <QFileDialog>
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

void VjoyCalibrationWindow::on_openFile_clicked()
{
    QMessageBox mb(this);
   QString fileName = QFileDialog::getOpenFileName(this, tr("Open Config"), "", tr("Config Files (*.yaml *.xml *.txt)"));
   mb.setText(fileName);
   mb.exec();
}
