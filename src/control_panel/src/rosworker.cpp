#include "control_panel/rosworker.h"

using namespace std::chrono_literals;

RosWorker::RosWorker(QObject *parent)
    : QThread(parent)
{
    // 确保上下文存在
    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }

    // 1. 创建节点（每次 New Worker 都会创建一个全新的 Node）
    // 为了防止重名警告，可以在名字后面加个随机数，或者 reliance on strict cleanup
    node_ = rclcpp::Node::make_shared("control_panel_client");

    // 2. 创建执行器
    executor_ = rclcpp::executors::SingleThreadedExecutor::make_shared();
    executor_->add_node(node_);

    // 3. 创建客户端
    client_transform_ = std::make_shared<rclcpp::AsyncParametersClient>(node_, "transform_node");
    client_inference_ = std::make_shared<rclcpp::AsyncParametersClient>(node_, "inference_node");
}

RosWorker::~RosWorker()
{
    stop();
}

void RosWorker::stop()
{
    if (executor_) {
        // 取消 spin，让 run() 函数返回
        executor_->cancel();
    }
}

void RosWorker::run()
{
    if (executor_) {
        // 使用 executor 的 spin，它支持被 cancel
        executor_->spin();
    }
}

void RosWorker::setParam(const std::string &name, double value)
{
    if (!client_inference_ || !client_transform_) return;

    auto send_param = [&](rclcpp::AsyncParametersClient::SharedPtr client, 
                          const std::string &target_node_name, 
                          const std::string &param_name, 
                          const rclcpp::Parameter &param_value) {
        
        // 缩短等待时间，避免界面卡太久
        if (!client->wait_for_service(500ms)) {
            return;
        }

        client->set_parameters({param_value}, 
            [](std::shared_future<std::vector<rcl_interfaces::msg::SetParametersResult>>){});
    };

    if (name == "show_image") {
        send_param(client_inference_, "inference_node", "show_image", rclcpp::Parameter("show_image", (value > 0.5)));
    }
    else if (name == "conf_thres") {
        send_param(client_inference_, "inference_node", "conf_thres", rclcpp::Parameter("conf_thres", value));
    }
    else if (name == "x_offset" || name == "y_offset" || name == "z_offset") {
        send_param(client_transform_, "transform_node", name, rclcpp::Parameter(name, value));
    }
}