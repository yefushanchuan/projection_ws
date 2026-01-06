#include "rclcpp/rclcpp.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp" 
#include "string"
#include <memory>

class CoordTransformNode : public rclcpp::Node
{
public:
    CoordTransformNode()
    : Node("coord_transformer_node")
    {
        this->declare_parameter<double>("offset_d", 0.05);
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
    void array_callback(const object3d_msgs::msg::Object3DArray::SharedPtr msg)
    {
        double offset_d = this->get_parameter("offset_d").as_double();
        object3d_msgs::msg::Object3DArray projected_msg;
        projected_msg.header = msg->header;
        projected_msg.header.frame_id = "projector_frame";

        for (const auto & input_obj : msg->objects) {
            auto output_obj = input_obj; // 复制对象

            output_obj.point.y = input_obj.point.y + offset_d; 
            
            projected_msg.objects.push_back(output_obj);
        }
        
        publisher_->publish(projected_msg);
        RCLCPP_INFO(this->get_logger(), "Transformed frame with %zu objects", projected_msg.objects.size());
    }
    rclcpp::Subscription<object3d_msgs::msg::Object3DArray>::SharedPtr subscription_;
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr publisher_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CoordTransformNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}