#include "datacapture.h"
#include "ui_datacapture.h"

datacapture::datacapture(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::datacapture)
{
    ui->setupUi(this);
}

datacapture::~datacapture()
{
    delete ui;
}
