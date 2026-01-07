#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QThread>
#include <QProcess>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QLabel>
#include <QVBoxLayout>
#include <QGroupBox>
#include <rclcpp/rclcpp.hpp>
#include <QCheckBox>

// ==========================================
// ROS 工作线程 (负责 spin，防止界面卡死)
// ==========================================
class RosWorker : public QThread {
    Q_OBJECT
public:
    void run() override {
        // 创建一个简单的节点用于发送参数请求
        node = rclcpp::Node::make_shared("gui_client_node");
        
        // 创建参数客户端，连接到 "/transform_node"
        param_client = std::make_shared<rclcpp::AsyncParametersClient>(node, "/transform_node");
        
        // 开启循环
        rclcpp::spin(node);
    }

    void setParam(const std::string& name, double value) {
        if (!param_client) return;
        // 异步设置参数，非阻塞，性能极高
        param_client->set_parameters({
            rclcpp::Parameter(name, value)
        });
    }

    rclcpp::Node::SharedPtr node;
    std::shared_ptr<rclcpp::AsyncParametersClient> param_client;
};

// ==========================================
// 主窗口
// ==========================================
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onStartClicked();
    void onStopClicked();
    // 统一处理参数变化的槽函数
    void onParamChanged(const QString &name, double value);

private:
    void setupUi(); // 手写 UI 布局，免去 .ui 文件烦恼

    // ROS 线程
    RosWorker *ros_worker;
    
    // Launch 进程
    QProcess *launch_process;

    // UI 控件
    QCheckBox *chk_show_image;
    QPushButton *btn_start;
    QPushButton *btn_stop;
    QDoubleSpinBox *spin_x;
    QDoubleSpinBox *spin_y;
    QDoubleSpinBox *spin_z;
};

#endif // MAINWINDOW_H