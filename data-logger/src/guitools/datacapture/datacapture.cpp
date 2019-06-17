#include "datacapture.h"
#include "ui_datacapture.h"
#include <QFileDialog>
#include <QMessageBox>
datacapture::datacapture(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::datacapture),
	configKeys({"search_string" , "images_folder", "udp_folder" , "driver_name" , "track_name" ,
		 "udp_threads" , "image_threads" , "image_capture_frequency" , "initial_delay_time" })
{
    ui->setupUi(this);
	ui->configTable->setRowCount(configKeys.size());
	for (unsigned int i = 0; i < configKeys.size(); i++)
	{
		ui->configTable->setItem(i, 0, new QTableWidgetItem(QString(configKeys.at(i).c_str())));
		ui->configTable->setItem(i, 1, new QTableWidgetItem(QString("")));
	}
}

datacapture::~datacapture()
{
    delete ui;
}

void datacapture::on_comboBox_activated(const QString &arg1)
{
}


void datacapture::on_browseConfigButton_clicked()
{


    QString fileName = QFileDialog::getOpenFileName(this,  "Open Config File" ,  "",   "Yaml Files (*.yaml *.yml)"  );
    ui->fileNameBox->setCurrentText(fileName);
    ui->fileNameBox->addItem(fileName);
}

void datacapture::on_loadConfigButton_clicked()
{
	std::string config_file = ui->fileNameBox->currentText().toStdString();
	if (config_file.empty())
	{
		return;
	}
	
	try {
		config_node = YAML::LoadFile(config_file);
	}
	catch (YAML::ParserException& e)
	{
		QMessageBox msgBox;
		std::stringstream ss;
		ss << "Could not parse provided file. Inner exception: " << std::endl;
		ss << e.what() << std::endl;
		msgBox.setText(QString(ss.str().c_str()));
		msgBox.exec();
		return;
	}
	catch (YAML::BadFile& e)
	{
		QMessageBox msgBox;
		std::stringstream ss;
		ss << "Provided file does not exist. " << std::endl;
		msgBox.setText(QString(ss.str().c_str()));
		msgBox.exec();
		return;
	}
	catch (std::exception& e)
	{
		QMessageBox msgBox;
		std::stringstream ss;
		ss << "Unknown exception when loading config file. Inner exception: " << std::endl;
		ss << e.what() << std::endl;
		msgBox.setText(QString(ss.str().c_str()));
		msgBox.exec();
		return;
	}
	for (unsigned int i = 0; i < configKeys.size(); i++)
	{
		ui->configTable->setItem(i, 0, new QTableWidgetItem(QString(configKeys.at(i).c_str())));
		try
		{
			ui->configTable->setItem(i, 1, new QTableWidgetItem(QString(config_node[configKeys.at(i)].as<std::string>().c_str())));
		}
		catch (std::exception& e)
		{
			ui->configTable->setItem(i, 1, new QTableWidgetItem(QString("")));

		}
	}
}

void datacapture::on_configTable_itemChanged(QTableWidgetItem *item)
{
	unsigned int row = item->row();
	unsigned int col = item->column();
	if (col == 0)
	{
		item->setText(QString(configKeys.at(row).c_str()));
	}

	//QMessageBox msgBox;
	//std::stringstream ss;
	//ss << "Changed value to " << item->text().toStdString();
	//msgBox.setText(QString(ss.str().c_str()));
	//msgBox.exec();
}

