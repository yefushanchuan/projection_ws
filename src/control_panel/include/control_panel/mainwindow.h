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

// 包含 worker 头文件
#include "control_panel/rosworker.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onStartClicked();
    void onStopClicked();
    void onParamChanged(const QString &name, double value);    // 统一处理参数变化的槽

private:
    void setupUi();

    // 界面控件
    QProcess *launch_process;
    QCheckBox *chk_show_image;
    QPushButton *btn_start;
    QPushButton *btn_stop;
    QDoubleSpinBox *spin_x, *spin_y, *spin_z;
    QLineEdit *le_model_path;
    QPushButton *btn_browse;

    RosWorker *ros_worker;    // ROS 工作线程
};

#endif // MAINWINDOW_H