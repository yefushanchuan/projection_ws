#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QCheckBox>
#include <QPushButton>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QDebug>
#include <QLineEdit>
#include <QFileDialog> 
#include <QStatusBar>
#include <QMessageBox>
#include <QCloseEvent>
#include <QApplication>

#include <rclcpp/rclcpp.hpp>
#include "control_panel/rosworker.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onStartClicked();
    void onStopClicked();
    void onParamChanged(const QString &name, double value);

private:
    void setupUi();
    
    void forceCleanUp(); 

    QProcess *launch_process;
    RosWorker *ros_worker;
    
    QStatusBar *statusBar;
    QLabel *statusLabel;
    
    QCheckBox *chk_show_image;
    QPushButton *btn_start;
    QPushButton *btn_stop;
    
    QDoubleSpinBox *spin_x;
    QDoubleSpinBox *spin_y;
    QDoubleSpinBox *spin_z;
    
    QLineEdit *le_model_path;
    QPushButton *btn_browse;
};

#endif // MAINWINDOW_H