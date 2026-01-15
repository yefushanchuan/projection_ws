#ifndef bpu_seg_dnn_H
#define bpu_seg_dnn_H

#include "yolo_seg_common.h"
#include <vector>
#include <string>

namespace yolo_seg {

// 配置结构体
struct Config {
    int class_num = 80;
    int reg_max = 16;
    int num_mask = 32;
    std::vector<int> strides = {8, 16, 32};
    float score_thres = 0.25;
    float nms_thres = 0.45;
    std::vector<std::string> class_names;
};

class BPU_Segment {
public:
    BPU_Segment();
    ~BPU_Segment() = default;

    // 1. 预处理：对应 Python 的 PreProcess
    // 将 BGR 图像 Resize 并转换为 NV12，供推理使用
    void PreProcess(const cv::Mat& bgr_img, 
                    int model_w, int model_h, 
                    cv::Mat& nv12_out);

    // 2. 后处理：对应 Python 的 PostProcess
    // 解析 Tensor 输出，生成结果
    void PostProcess(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
                     int model_h, int model_w,
                     std::vector<SegResult>& results);

    // 3. 可视化：对应 Python 的 detect_result / draw_detection
    void Visualize(cv::Mat& img, 
                   const std::vector<SegResult>& results, 
                   int model_w, int model_h, 
                   double fps,
                   bool show_window = true);

private:
    Config config_;
    bool window_created_ = false;

    // 内部辅助函数
    cv::Scalar GetColor(int id);
    void BGRToNV12(const cv::Mat& bgr, cv::Mat& nv12);
};

} // namespace yolo_seg

#endif // bpu_seg_dnn_H