#ifndef SEGMENT_YOLO_SEG_NODE_H
#define SEGMENT_YOLO_SEG_NODE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <opencv2/opencv.hpp>
#include "dnn_node/dnn_node_data.h"

namespace yolo_seg {

struct SegResult {
    int id;
    float score;
    cv::Rect box;
    std::string class_name;
    cv::Mat mask; 
};

struct Config {
    int class_num = 80;
    int reg_max = 16;
    int num_mask = 32;
    std::vector<int> strides = {8, 16, 32};
    float score_thres = 0.25;
    float nms_thres = 0.45;
    std::vector<std::string> class_names; 
};

inline float fastExp(float x) {
    union { uint32_t i; float f; } v;
    v.i = (12102203.1616540672f * x + 1064807160.56887296f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

class Parser {
public:
    Config config_;

    Parser() {
        config_.class_names.resize(80, "object");
        // 这里只是示例，建议替换为真实的 COCO 类别表
        config_.class_names[0] = "person";
    }

    void Parse(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
               int model_h, int model_w,
               std::vector<SegResult>& results) {
        
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> mask_coeffs;

        // 1. 遍历 3 个检测头 (Stride 8, 16, 32)
        for (int i = 0; i < 3; ++i) {
            if ((size_t)(i * 3 + 2) >= tensors.size()) break;

            int stride = config_.strides[i];
            auto& cls_tensor = tensors[i * 3];
            auto& box_tensor = tensors[i * 3 + 1];
            auto& msk_tensor = tensors[i * 3 + 2];
            
            float* cls_data = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
            float* box_data = reinterpret_cast<float*>(box_tensor->sysMem[0].virAddr);
            float* msk_data = reinterpret_cast<float*>(msk_tensor->sysMem[0].virAddr);

            // =========================================================
            // !!! 修复核心：适配 NHWC 布局的维度索引 !!!
            // [0]:N, [1]:H, [2]:W, [3]:C
            // =========================================================
            int h = cls_tensor->properties.validShape.dimensionSize[1]; 
            int w = cls_tensor->properties.validShape.dimensionSize[2]; 
            
            // 步长
            int cls_step = config_.class_num;       // 80
            int box_step = 4 * config_.reg_max;     // 64
            int msk_step = config_.num_mask;        // 32

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int offset = (y * w + x); // 空间偏移量
                    
                    // 1. 找最大类别
                    float max_score = -1000.0f;
                    int max_id = -1;
                    
                    // 指针移动：NHWC 模式下，同一个像素点的不同通道是连续的
                    float* cur_cls = cls_data + offset * cls_step;
                    
                    for(int c=0; c<config_.class_num; ++c) {
                        if(cur_cls[c] > max_score) {
                            max_score = cur_cls[c];
                            max_id = c;
                        }
                    }
                    
                    float score = sigmoid(max_score);
                    if (score < config_.score_thres) continue;

                    // 2. 解码 Box
                    float* cur_box = box_data + offset * box_step;
                    float dist[4] = {0};
                    for(int k=0; k<4; ++k) {
                        float val = 0;
                        float sum = 0;
                        for(int r=0; r<config_.reg_max; ++r) {
                            float e = fastExp(cur_box[k*config_.reg_max + r]);
                            val += e * r;
                            sum += e;
                        }
                        dist[k] = val / sum;
                    }
                    
                    float x1 = (x + 0.5f - dist[0]) * stride;
                    float y1 = (y + 0.5f - dist[1]) * stride;
                    float x2 = (x + 0.5f + dist[2]) * stride;
                    float y2 = (y + 0.5f + dist[3]) * stride;

                    // 3. 读取 Mask 系数
                    float* cur_msk = msk_data + offset * msk_step;
                    std::vector<float> coeffs(cur_msk, cur_msk + config_.num_mask);

                    class_ids.push_back(max_id);
                    confidences.push_back(score);
                    boxes.emplace_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                    mask_coeffs.push_back(coeffs);
                }
            }
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, config_.score_thres, config_.nms_thres, indices);

        if (indices.empty()) return;

        // 2. 处理 Proto Mask
        auto& proto_tensor = tensors.back();
        float* proto_ptr = reinterpret_cast<float*>(proto_tensor->sysMem[0].virAddr);
        
        // Log 显示 Proto 是 [1, 160, 160, 32] (NCHW标签但实际上可能是HWC数据)
        // 关键点：OpenCV 的 Mat 是 HWC 排列的。
        // 如果 BPU 输出是 NHWC (160, 160, 32)，那么它是 packed 的 (RGBRGB...)
        // 我们需要把它整理成矩阵乘法适合的形状。
        
        int proto_h = 160;
        int proto_w = 160;
        int proto_c = 32;

        // 将 Proto 视为 H*W 行，C 列的矩阵 (25600, 32)
        // 注意：这假设内存布局是 NHWC (packed)。如果结果不对，可能是 NCHW (planar)
        cv::Mat proto_mat(proto_h * proto_w, proto_c, CV_32F, proto_ptr);

        for (int idx : indices) {
            SegResult res;
            res.id = class_ids[idx];
            res.score = confidences[idx];
            res.box = boxes[idx];
            res.class_name = ((size_t)res.id < config_.class_names.size()) ? config_.class_names[res.id] : "unknown";

            // Mask 计算: (1, 32) * (32, 25600) -> 矩阵乘法
            // 但我们的 proto_mat 是 (25600, 32)，所以我们用 coeff * proto_t 或者 proto * coeff_t
            
            cv::Mat coeff_mat(1, proto_c, CV_32F, mask_coeffs[idx].data());
            
            // 计算 mask = proto_mat * coeff_mat.t()
            // (25600, 32) * (32, 1) -> (25600, 1)
            cv::Mat mask_raw = proto_mat * coeff_mat.t();
            
            // Reshape 为 (160, 160)
            mask_raw = mask_raw.reshape(1, proto_h);
            
            // Sigmoid
            cv::exp(-mask_raw, mask_raw);
            mask_raw = 1.0 / (1.0 + mask_raw);
            
            // Resize
            cv::Mat mask_large;
            cv::resize(mask_raw, mask_large, cv::Size(model_w, model_h));
            
            // Crop & Binarize
            cv::Mat final_mask = cv::Mat::zeros(model_h, model_w, CV_8UC1);
            cv::Mat mask_binary = mask_large > 0.5;
            
            cv::Rect valid_box = res.box & cv::Rect(0, 0, model_w, model_h);
            if (valid_box.area() > 0) {
                mask_binary(valid_box).copyTo(final_mask(valid_box));
            }
            
            res.mask = final_mask;
            results.push_back(res);
        }
    }
};

} // namespace yolo_seg

#endif