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

// 定义结果结构体
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
        // 简单示例，实际请填充完整列表
        config_.class_names[0] = "person";
    }

    void Parse(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
               int model_h, int model_w,
               std::vector<SegResult>& results) {
        
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> mask_coeffs;

        for (int i = 0; i < 3; ++i) {
            // 类型转换修正：显式转换 size() 避免警告
            if ((size_t)(i * 3 + 2) >= tensors.size()) break;

            int stride = config_.strides[i];
            auto& cls_tensor = tensors[i * 3];
            auto& box_tensor = tensors[i * 3 + 1];
            auto& msk_tensor = tensors[i * 3 + 2];
            
            float* cls_data = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
            float* box_data = reinterpret_cast<float*>(box_tensor->sysMem[0].virAddr);
            float* msk_data = reinterpret_cast<float*>(msk_tensor->sysMem[0].virAddr);

            int h = cls_tensor->properties.validShape.dimensionSize[2]; 
            int w = cls_tensor->properties.validShape.dimensionSize[3]; 
            
            int cls_step = config_.class_num;
            int box_step = 4 * config_.reg_max;
            int msk_step = config_.num_mask;

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int offset = (y * w + x); 
                    float max_score = -1000.0f;
                    int max_id = -1;
                    float* cur_cls = cls_data + offset * cls_step;
                    
                    for(int c=0; c<config_.class_num; ++c) {
                        if(cur_cls[c] > max_score) {
                            max_score = cur_cls[c];
                            max_id = c;
                        }
                    }
                    
                    float score = sigmoid(max_score);
                    if (score < config_.score_thres) continue;

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

                    float* cur_msk = msk_data + offset * msk_step;
                    std::vector<float> coeffs(cur_msk, cur_msk + config_.num_mask);

                    class_ids.push_back(max_id);
                    confidences.push_back(score);
                    boxes.emplace_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                    mask_coeffs.push_back(coeffs);
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, config_.score_thres, config_.nms_thres, indices);

        if (indices.empty()) return;

        auto& proto_tensor = tensors.back();
        float* proto_ptr = reinterpret_cast<float*>(proto_tensor->sysMem[0].virAddr);
        int proto_c = 32;
        int proto_h = 160;
        int proto_w = 160;
        cv::Mat proto_mat(proto_c, proto_h * proto_w, CV_32F, proto_ptr);

        for (int idx : indices) {
            SegResult res;
            res.id = class_ids[idx];
            res.score = confidences[idx];
            res.box = boxes[idx];
            // 类型转换修正
            res.class_name = ((size_t)res.id < config_.class_names.size()) ? config_.class_names[res.id] : "unknown";

            cv::Mat coeff_mat(1, proto_c, CV_32F, mask_coeffs[idx].data());
            cv::Mat mask_raw = coeff_mat * proto_mat;
            mask_raw = mask_raw.reshape(1, proto_h);
            cv::exp(-mask_raw, mask_raw);
            mask_raw = 1.0 / (1.0 + mask_raw);
            
            cv::Mat mask_large;
            cv::resize(mask_raw, mask_large, cv::Size(model_w, model_h));
            
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