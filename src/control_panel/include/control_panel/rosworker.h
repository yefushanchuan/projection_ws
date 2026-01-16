#ifndef ROSWORKER_H
#define ROSWORKER_H

#include <QThread>
#include <rclcpp/rclcpp.hpp>

class RosWorker : public QThread
{
    Q_OBJECT // 必须有这个宏

public:
    explicit RosWorker(QObject *parent = nullptr);
    ~RosWorker() override;

    void run() override;    // 线程入口，start() 后会自动调用

    void setParam(const std::string &name, double value);    // 设置参数的接口

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::AsyncParametersClient::SharedPtr client_transform_; // 用于 offset
    rclcpp::AsyncParametersClient::SharedPtr client_inference_;    // 用于 show_image
};

#endif // ROSWORKER_H