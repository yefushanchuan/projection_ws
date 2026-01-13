#include <QApplication>
#include <rclcpp/rclcpp.hpp>
#include "control_panel/mainwindow.h"

int main(int argc, char *argv[])
{
    // 1. 全局初始化 ROS / 初始化 Qt 应用
    rclcpp::init(argc, argv);
    QApplication a(argc, argv);

    // 2. 创建主窗口
    MainWindow w;
    w.show();

    // 3. 进入 Qt 事件循环
    int ret = a.exec();

    // 4. 确保退出时 ROS 已关闭 (双重保险)
    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }

    return ret;
}