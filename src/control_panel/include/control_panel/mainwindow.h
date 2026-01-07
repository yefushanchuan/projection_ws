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
    // 统一处理参数变化的槽
    void onParamChanged(const QString &name, double value);

private:
    void setupUi();

    // 界面控件
    QProcess *launch_process;
    QCheckBox *chk_show_image;
    QPushButton *btn_start;
    QPushButton *btn_stop;
    QDoubleSpinBox *spin_x, *spin_y, *spin_z;

    // ROS 工作线程
    RosWorker *ros_worker;
};

#endif // MAINWINDOW_H