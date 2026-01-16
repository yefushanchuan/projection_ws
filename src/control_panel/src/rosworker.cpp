#include "control_panel/rosworker.h"

RosWorker::RosWorker(QObject *parent)
    : QThread(parent)
{
    // 1. 创建本节点
    node_ = rclcpp::Node::make_shared("control_panel_node");

    // 2. 创建参数客户端
    // 客户端 A: 控制 XYZ 偏移 (C++节点: transform_node)
    client_transform_ = std::make_shared<rclcpp::AsyncParametersClient>(node_, "transform_node");
    
    // 客户端 B: 控制是否显示图像 (yolo_seg_node / yolo_detect_node)
    client_inference_ = std::make_shared<rclcpp::AsyncParametersClient>(node_, "inference_node");
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
    // ==========================================
    // 场景 1: 控制 show_image (布尔值)
    // ==========================================
    if (name == "show_image") {
        if (!client_inference_->service_is_ready()) {
            RCLCPP_WARN(node_->get_logger(), "Inference node not ready for show_image");
            return;
        }

        // 逻辑转换：Qt传递过来的是 1.0 或 0.0，转为 bool
        bool bool_val = (value > 0.5);

        client_inference_->set_parameters({rclcpp::Parameter("show_image", bool_val)});
    }
    
    // ==========================================
    // 场景 2: 控制 offset (浮点数)
    // ==========================================
    else if (name == "x_offset" || name == "y_offset" || name == "z_offset") {
        if (!client_transform_->service_is_ready()) {
             RCLCPP_WARN(node_->get_logger(), "Transform node not ready for offsets");
             return;
        }

        client_transform_->set_parameters({rclcpp::Parameter(name, value)});
    }
    else {
        RCLCPP_WARN(node_->get_logger(), "Unknown param received: %s", name.c_str());
    }
}