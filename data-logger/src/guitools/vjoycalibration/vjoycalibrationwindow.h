#ifndef VJOYCALIBRATIONWINDOW_H
#define VJOYCALIBRATIONWINDOW_H

#include <QMainWindow>

namespace Ui {
class VjoyCalibrationWindow;
}

class VjoyCalibrationWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit VjoyCalibrationWindow(QWidget *parent = nullptr);
    ~VjoyCalibrationWindow();

private slots:

    void on_openFile_clicked();

private:
    Ui::VjoyCalibrationWindow *ui;
};

#endif // VJOYCALIBRATIONWINDOW_H