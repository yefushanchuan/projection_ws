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

    // 线程入口，start() 后会自动调用
    void run() override;

    // 设置参数的接口
    void setParam(const std::string &name, double value);

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::AsyncParametersClient::SharedPtr param_client_;
};

#endif // ROSWORKER_H