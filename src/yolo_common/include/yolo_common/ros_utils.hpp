#pragma once
#include "types.hpp"
#include "math_utils.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp"
#include <vector>

namespace yolo_common {
namespace ros_utils {

// 定义一个简单的相机内参结构体，方便传参
struct CameraIntrinsics {
    double fx, fy, cx, cy;
};

/**
 * @brief 将算法结果 + 深度图 转换为 ROS 3D 消息数组
 * @param results 算法输出的统一结果列表
 * @param depth_img 对齐后的 16位 深度图
 * @param header 原始图像的消息头 (用于时间戳对齐)
 * @param cam 内参
 * @param msg_out [输出] 填充好的 ROS 消息
 */
inline void ResultsTo3DMessage(const std::vector<UnifiedResult>& results,
                               const cv::Mat& depth_img,
                               const std_msgs::msg::Header& header,
                               const CameraIntrinsics& cam,
                               object3d_msgs::msg::Object3DArray& msg_out) 
{
    msg_out.header = header;
    msg_out.objects.clear();

    for (const auto& res : results) {
        // 1. 获取深度 (使用 math_utils)
        float z_m = yolo_common::math::GetRobustDepth(depth_img, res.center);
        if (z_m <= 0.0f) continue;

        // 2. 3D 投影 (使用 math_utils)
        auto p3d = yolo_common::math::ProjectPixelTo3D(res.center.x, res.center.y, z_m, 
                                                       cam.fx, cam.fy, cam.cx, cam.cy);

        // 3. 计算物理尺寸 (检测框或最小外接圆的投影)
        float width_m = 0.0f;
        float height_m = 0.0f;

        if (res.mic_radius > 0) {
            // 分割模式：基于内切圆半径
            float r_m = (res.mic_radius * z_m) / (float)cam.fx;
            width_m = r_m * 2.0f;
            height_m = r_m * 2.0f;
        } else {
            // 检测模式：基于 Box 宽高
            width_m = (res.box.width * z_m) / (float)cam.fx;
            height_m = (res.box.height * z_m) / (float)cam.fy;
        }

        // 4. 填充消息
        object3d_msgs::msg::Object3D obj;
        obj.class_name = res.class_name;
        obj.score = res.score;
        obj.point.x = p3d.x;
        obj.point.y = p3d.y;
        obj.point.z = p3d.z;
        obj.width_m = width_m;
        obj.height_m = height_m;

        msg_out.objects.push_back(obj);
    }
}

} // namespace ros_utils
} // namespace yolo_common