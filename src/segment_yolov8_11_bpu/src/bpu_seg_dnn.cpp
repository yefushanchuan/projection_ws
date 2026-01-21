#include "segment_yolov8_11_bpu/bpu_seg_dnn.hpp"
#include "yolo_common/math_utils.hpp"
#include "yolo_common/class_names.hpp"
#include "yolo_common/img_proc.hpp"

BPU_Segment::BPU_Segment() {
    config_.class_names = yolo_common::COCO_CLASSES;
    config_.class_num = config_.class_names.size();
}

void BPU_Segment::BGRToNV12(const cv::Mat& bgr, cv::Mat& nv12) {
    int w = bgr.cols;
    int h = bgr.rows;
    cv::Mat yuv_i420;
    cv::cvtColor(bgr, yuv_i420, cv::COLOR_BGR2YUV_I420);
    nv12.create(h * 1.5, w, CV_8UC1);
    memcpy(nv12.data, yuv_i420.data, w * h);
    uint8_t* u_ptr = yuv_i420.data + w * h;
    uint8_t* v_ptr = u_ptr + (w * h) / 4;
    uint8_t* uv_ptr = nv12.data + w * h;
    for (int i = 0; i < (w * h) / 4; ++i) {
        *uv_ptr++ = *u_ptr++;
        *uv_ptr++ = *v_ptr++;
    }
}

void BPU_Segment::PreProcess(const cv::Mat& bgr_img, int model_w, int model_h, cv::Mat& nv12_out) {
    cv::Mat letterboxed;
    ratio_ = yolo_common::proc::Letterbox(bgr_img, letterboxed, model_w, model_h, pad_w_, pad_h_);
    BGRToNV12(letterboxed, nv12_out);
}

void BPU_Segment::PostProcess(const std::vector<std::shared_ptr<hobot::dnn_node::DNNTensor>>& tensors,
                             int model_h, int model_w,
                             std::vector<yolo_common::UnifiedResult>& results) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> mask_coeffs;

    // 1. 解析 Head
    for (int i = 0; i < 3; ++i) {
        if ((size_t)(i * 3 + 2) >= tensors.size()) break;
        int stride = config_.strides[i];
        
        auto& cls_tensor = tensors[i * 3];
        auto& box_tensor = tensors[i * 3 + 1];
        auto& msk_tensor = tensors[i * 3 + 2];
        
        float* cls_data = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
        float* box_data = reinterpret_cast<float*>(box_tensor->sysMem[0].virAddr);
        float* msk_data = reinterpret_cast<float*>(msk_tensor->sysMem[0].virAddr);

        int h_dim = cls_tensor->properties.validShape.dimensionSize[1]; 
        int w_dim = cls_tensor->properties.validShape.dimensionSize[2]; 

        int cls_step = config_.class_num;
        int box_step = 4 * config_.reg_max;
        int msk_step = config_.num_mask;

        for (int y = 0; y < h_dim; ++y) {
            for (int x = 0; x < w_dim; ++x) {
                int offset = (y * w_dim + x);
                
                // Classify
                float max_score = -1000.0f;
                int max_id = -1;
                float* cur_cls = cls_data + offset * cls_step;
                for(int c=0; c<config_.class_num; ++c) {
                    if(cur_cls[c] > max_score) { max_score = cur_cls[c]; max_id = c; }
                }
                
                // 使用 yolo_common::math::sigmoid
                float score = yolo_common::math::sigmoid(max_score);
                if (score < config_.conf_thres) continue;

                // Box Decode
                float* cur_box = box_data + offset * box_step;
                float dist[4];
                for(int k=0; k<4; ++k) {
                    float val = 0, sum = 0;
                    for(int r=0; r<config_.reg_max; ++r) {
                        // 使用 yolo_common::math::fastExp
                        float e = yolo_common::math::fastExp(cur_box[k*config_.reg_max + r]);
                        val += e * r; sum += e;
                    }
                    dist[k] = val / sum;
                }
                
                float pb_cx = (x + 0.5f) * stride;
                float pb_cy = (y + 0.5f) * stride;
                float x1 = pb_cx - dist[0] * stride;
                float y1 = pb_cy - dist[1] * stride;
                float x2 = pb_cx + dist[2] * stride;
                float y2 = pb_cy + dist[3] * stride;

                // Mask Coeffs
                float* cur_msk = msk_data + offset * msk_step;
                std::vector<float> coeffs(cur_msk, cur_msk + config_.num_mask);

                class_ids.push_back(max_id);
                confidences.push_back(score);
                boxes.emplace_back(cv::Rect(x1, y1, x2-x1, y2-y1));
                mask_coeffs.push_back(coeffs);
            }
        }
    }

    // 2. NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, config_.conf_thres, config_.nms_thres, indices);
    if (indices.empty()) return;

    // 3. Process Proto Mask
    auto& proto_tensor = tensors.back();
    float* proto_ptr = reinterpret_cast<float*>(proto_tensor->sysMem[0].virAddr);
    int proto_h = 160; 
    int proto_w = 160;
    int proto_c = 32;
    cv::Mat proto_mat(proto_h * proto_w, proto_c, CV_32F, proto_ptr);

    for (int idx : indices) {
        yolo_common::UnifiedResult res; // 使用 UnifiedResult
        res.id = class_ids[idx];
        res.score = confidences[idx];
        res.class_name = (res.id < (int)config_.class_names.size()) ? config_.class_names[res.id] : "unknown";

        // === 1. 获取模型尺度下的 Box ===
        cv::Rect model_box = boxes[idx];
        model_box = model_box & cv::Rect(0, 0, model_w, model_h);
        if (model_box.area() <= 0) continue;

        // === 2. 还原最终坐标 (x - pad) / ratio ===
        float bx1 = (model_box.x - pad_w_) / ratio_;
        float by1 = (model_box.y - pad_h_) / ratio_;
        float bx2 = (model_box.br().x - pad_w_) / ratio_;
        float by2 = (model_box.br().y - pad_h_) / ratio_;
        res.box = cv::Rect(cv::Point(bx1, by1), cv::Point(bx2, by2));

        // === 3. Mask 计算与裁切 ===
        cv::Mat coeff_mat(1, proto_c, CV_32F, mask_coeffs[idx].data());
        cv::Mat mask_raw = proto_mat * coeff_mat.t(); // (25600 x 1)
        mask_raw = mask_raw.reshape(1, proto_h);      // 160x160
        
        // Sigmoid
        cv::exp(-mask_raw, mask_raw);
        mask_raw = 1.0 / (1.0 + mask_raw); 

        // Resize & Crop
        cv::Mat mask_model_full;
        cv::resize(mask_raw, mask_model_full, cv::Size(model_w, model_h));
        
        // 核心：保存裁剪后的掩码
        res.mask = mask_model_full(model_box).clone();

        // === 4. 计算最大内切圆 (MIC) ===
        if (!res.mask.empty()) {
            cv::Mat mask_bin;
            cv::threshold(res.mask, mask_bin, 0.5, 1.0, cv::THRESH_BINARY);
            mask_bin.convertTo(mask_bin, CV_8UC1);

            cv::Mat dist_map;
            cv::distanceTransform(mask_bin, dist_map, cv::DIST_L2, cv::DIST_MASK_PRECISE);
            
            double max_val;
            cv::Point max_loc; // Relative to model_box
            cv::minMaxLoc(dist_map, nullptr, &max_val, nullptr, &max_loc);

            // 映射回原图尺寸
            res.mic_radius = (float)max_val / ratio_;

            float center_x = (model_box.x + max_loc.x - pad_w_) / ratio_;
            float center_y = (model_box.y + max_loc.y - pad_h_) / ratio_;
            
            // UnifiedResult 使用 center 字段
            res.center = cv::Point2f(center_x, center_y);
        } else {
            res.mic_radius = 0;
            res.center = cv::Point2f(res.box.x + res.box.width/2, res.box.y + res.box.height/2);
        }

        results.push_back(res);
    }
}