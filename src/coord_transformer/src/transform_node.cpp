#include "rclcpp/rclcpp.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp" 
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "string"
#include <memory>

class CoordTransformNode : public rclcpp::Node
{
public:
    CoordTransformNode()
    : Node("transformer_node")
    {
        this->declare_parameter<double>("x_offset", 0.0);
        this->declare_parameter<double>("y_offset", 0.0);
        this->declare_parameter<double>("z_offset", 0.0);

        x_off_ = this->get_parameter("x_offset").as_double();
        y_off_ = this->get_parameter("y_offset").as_double();
        z_off_ = this->get_parameter("z_offset").as_double();

        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&CoordTransformNode::parameters_callback, this, std::placeholders::_1));

        subscription_ = this->create_subscription<object3d_msgs::msg::Object3DArray>(
            "target_points_array",
            rclcpp::QoS(10),
            std::bind(&CoordTransformNode::array_callback, this, std::placeholders::_1)
        );
        publisher_ = this->create_publisher<object3d_msgs::msg::Object3DArray>(
            "target_points_projection_array",
            rclcpp::QoS(10)
        );
        RCLCPP_INFO(this->get_logger(), "CoordTransformNode has been started.");
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
                x_off_ = param.as_double(); // 更新缓存变量
            }
            else if (param.get_name() == "y_offset") {
                y_off_ = param.as_double();
            }
            else if (param.get_name() == "z_offset") {
                z_off_ = param.as_double();
            }
        }
        return result;
    }
    
    void array_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
    {
        object3d_msgs::msg::Object3DArray projected_msg;
        projected_msg.header = msg->header;
        projected_msg.header.frame_id = "projector_frame";

        for (const auto & input_obj : msg->objects) {
            auto output_obj = input_obj; // 复制对象

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
    auto node = std::make_shared<CoordTransformNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}