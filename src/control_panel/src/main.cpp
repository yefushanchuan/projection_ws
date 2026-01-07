#include "control_panel/mainwindow.h"
#include <QApplication>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char *argv[])
{
    // 1. 初始化 ROS
    rclcpp::init(argc, argv);

    // 2. 初始化 Qt
    QApplication a(argc, argv);
    
    MainWindow w;
    w.resize(400, 300);
    w.show();

    // 3. 进入 Qt 事件循环
    return a.exec();
}