#ifndef DATACAPTURE_H
#define DATACAPTURE_H

#include <QMainWindow>

namespace Ui {
class datacapture;
}

class datacapture : public QMainWindow
{
    Q_OBJECT

public:
    explicit datacapture(QWidget *parent = nullptr);
    ~datacapture();

private:
    Ui::datacapture *ui;
};

#endif // DATACAPTURE_H
