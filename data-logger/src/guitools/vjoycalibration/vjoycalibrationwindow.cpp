#include "vjoycalibrationwindow.h"
#include "ui_vjoycalibrationwindow.h"
#include <QMessageBox>
#include <QFileDialog>
VjoyCalibrationWindow::VjoyCalibrationWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::VjoyCalibrationWindow),
    scene(new QGraphicsScene)
{
    ui->setupUi(this);
}

VjoyCalibrationWindow::~VjoyCalibrationWindow()
{
    delete ui;
    delete scene;
}

void VjoyCalibrationWindow::on_openFile_clicked()
{
    QMessageBox mb(this);
   QString fileName = QFileDialog::getOpenFileName(this, tr("Open Config"), "", tr("Image Files (*.jpg *.png)"));
   mb.setText(fileName);
   mb.exec();
   QImage imageObject;
   imageObject.load(fileName);
   QPixmap image = QPixmap::fromImage(imageObject);
   delete scene;
   scene = new QGraphicsScene(this);
   scene->addPixmap(image);
   scene->setSceneRect(image.rect());
   ui->graphicsView->setScene(scene);
  // ui->graphicsView->
}
