#include "control_panel/rosworker.h"

RosWorker::RosWorker(QObject *parent)
    : QThread(parent)
{
    // 1. 创建本节点
    node_ = rclcpp::Node::make_shared("control_panel_node");

    // 2. 创建参数客户端
    // !!! 注意：这里必须填你要控制的那个节点的名称 !!!
    // 例如你的算法节点叫 "projection_node"
    std::string target_node_name = "projection_node";
    param_client_ = std::make_shared<rclcpp::AsyncParametersClient>(node_, target_node_name);
}

RosWorker::~RosWorker()
{
    // 析构时不需要手动做太多，主要在 MainWindow 里控制停止
}

void RosWorker::run()
{
    // 这是一个死循环，会在新线程里一直运行，直到 rclcpp::shutdown() 被调用
    rclcpp::spin(node_);
}

void RosWorker::setParam(const std::string &name, double value)
{
    // 简单的检查
    if (!param_client_->service_is_ready()) {
        RCLCPP_WARN(node_->get_logger(), "Target node not ready, skipping param set: %s", name.c_str());
        return;
    }

    // 异步发送参数修改请求
    param_client_->set_parameters(
        {rclcpp::Parameter(name, value)},
        [this, name](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>> future) {
            // 这里是回调，注意不要在这里直接操作 UI 控件
            (void)future; // 消除未使用变量警告
            RCLCPP_INFO(node_->get_logger(), "Param set request sent: %s", name.c_str());
        }
    );
}