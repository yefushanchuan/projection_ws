#pragma once
#include <cmath>
#include <algorithm>
#include <vector>
#include <opencv2/core.hpp>

namespace yolo_common {
namespace math {

// --- 来自 Code 1 的高性能数学函数 ---
inline float fastExp(float x) {
    union {
        uint32_t i;
        float f;
    } v;
    v.i = (12102203.1616540672f * x + 1064807160.56887296f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// --- 通用的深度提取逻辑 (5x5 中值滤波) ---
// 返回 -1.0f 表示无效
inline float GetRobustDepth(const cv::Mat& depth_img, const cv::Point2f& pt) {
    if (depth_img.empty()) return -1.0f;
    
    int cx = std::round(pt.x);
    int cy = std::round(pt.y);
    int w = depth_img.cols;
    int h = depth_img.rows;

    std::vector<uint16_t> valid_depths;
    valid_depths.reserve(25); 

    // 遍历 5x5 窗口
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int u = cx + dx;
            int v = cy + dy;
            if (u >= 0 && u < w && v >= 0 && v < h) {
                // 假设深度图是 16UC1 (单位 mm)
                uint16_t d = depth_img.at<uint16_t>(v, u);
                if (d > 0) {
                    valid_depths.push_back(d);
                }
            }
        }
    }

    if (valid_depths.empty()) return -1.0f;

    // 中值滤波
    size_t n = valid_depths.size() / 2;
    std::nth_element(valid_depths.begin(), valid_depths.begin() + n, valid_depths.end());
    
    // 转换为米 (m)
    return static_cast<float>(valid_depths[n]) / 1000.0f;
}

// --- 像素坐标反投影到 3D 点 ---
inline cv::Point3f ProjectPixelTo3D(float u, float v, float z_m, 
                                    double fx, double fy, double cx, double cy) {
    float x_m = (u - (float)cx) * z_m / (float)fx;
    float y_m = (v - (float)cy) * z_m / (float)fy;
    return cv::Point3f(x_m, y_m, z_m);
}

} // namespace math
} // namespace yolo_common