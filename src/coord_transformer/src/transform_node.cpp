#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp" 
#include "rcl_interfaces/msg/set_parameters_result.hpp"

class CoordTransformNode : public rclcpp::Node
{
public:
    CoordTransformNode()
    : Node("transform_node") // 注意：这里节点名最好和 Launch 文件里保持一致
    {
        // 声明参数
        this->declare_parameter<double>("x_offset", 0.0);
        this->declare_parameter<double>("y_offset", 0.0);
        this->declare_parameter<double>("z_offset", 0.0);

        // 初始化读取
        x_off_ = this->get_parameter("x_offset").as_double();
        y_off_ = this->get_parameter("y_offset").as_double();
        z_off_ = this->get_parameter("z_offset").as_double();

        // 注册参数回调
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&CoordTransformNode::parameters_callback, this, std::placeholders::_1));

        // 订阅原始坐标
        subscription_ = this->create_subscription<object3d_msgs::msg::Object3DArray>(
            "target_points_array",
            10,
            std::bind(&CoordTransformNode::array_callback, this, std::placeholders::_1)
        );
        
        // 发布转换后的坐标
        publisher_ = this->create_publisher<object3d_msgs::msg::Object3DArray>(
            "target_points_projection_array",
            10
        );
        
        RCLCPP_INFO(this->get_logger(), "CoordTransformNode started. Initial Offsets: X=%.2f, Y=%.2f, Z=%.2f", 
            x_off_, y_off_, z_off_);
    }

private:
    rcl_interfaces::msg::SetParametersResult parameters_callback(
        const std::vector<rclcpp::Parameter> & parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        for (const auto & param : parameters)
        {
            if (param.get_name() == "x_offset") {
                x_off_ = param.as_double();
                // 打印调试日志，方便在 Qt 拖动时看到反应
                RCLCPP_INFO(this->get_logger(), "Update X Offset: %.2f", x_off_);
            }
            else if (param.get_name() == "y_offset") {
                y_off_ = param.as_double();
                RCLCPP_INFO(this->get_logger(), "Update Y Offset: %.2f", y_off_);
            }
            else if (param.get_name() == "z_offset") {
                z_off_ = param.as_double();
                RCLCPP_INFO(this->get_logger(), "Update Z Offset: %.2f", z_off_);
            }
        }
        return result;
    }
    
    void array_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
    {
        object3d_msgs::msg::Object3DArray projected_msg;
        projected_msg.header = msg->header;
        // 修改 frame_id 以区分坐标系
        projected_msg.header.frame_id = "projector_frame";

        for (const auto & input_obj : msg->objects) {
            auto output_obj = input_obj; 

            // 执行偏移计算
            output_obj.point.x = input_obj.point.x + x_off_;
            output_obj.point.y = input_obj.point.y + y_off_;
            output_obj.point.z = input_obj.point.z + z_off_;
            
            projected_msg.objects.push_back(output_obj);
        }
        
        publisher_->publish(projected_msg);
    }

    rclcpp::Subscription<object3d_msgs::msg::Object3DArray>::SharedPtr subscription_;
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr publisher_;

    double x_off_, y_off_, z_off_;
    rclcpp::Node::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CoordTransformNode>());
    rclcpp::shutdown();
    return 0;
}