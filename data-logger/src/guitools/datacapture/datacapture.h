#ifndef DATACAPTURE_H
#define DATACAPTURE_H

#include <QMainWindow>
#include <QTableWidgetItem>
#include <memory>
#include <unordered_map>
#include <set>
#include <vector>
#include <yaml-cpp/yaml.h>
namespace Ui {
class datacapture;
}
class datacapture : public QMainWindow
{
    Q_OBJECT

public:
    explicit datacapture(QWidget *parent = nullptr);
    ~datacapture();

private slots:

    void on_comboBox_activated(const QString &arg1);

    void on_browseConfigButton_clicked();

    void on_loadConfigButton_clicked();

    void on_configTable_itemChanged(QTableWidgetItem *item);

    void on_imageLabel_linkActivated(const QString &link);

    void on_loadImageButton_clicked();

    void on_appList_indexesMoved(const QModelIndexList &indexes);

private:
     Ui::datacapture* ui;
	 YAML::Node config_node;
	 const std::vector<std::string> configKeys;
};

#endif // DATACAPTURE_H
