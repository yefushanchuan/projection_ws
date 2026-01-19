#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>

// ROS2 Includes
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "ament_index_cpp/get_package_share_directory.hpp" 

// Custom Message
#include "object3d_msgs/msg/object3_d.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp"

// OpenCV & ONNX Runtime
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std::chrono_literals;

// ================= COCO 类别定义 =================
const std::vector<std::string> COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// ================= 检测结果结构体 =================
struct Detection {
    int class_id;
    float confidence;
    cv::Rect box; // x, y, w, h
    cv::Point center;
};

// ================= YOLOv11 ONNX 推理类 =================
class Yolo11Detector {
public:
    Yolo11Detector(const std::string& model_path, float conf_thres = 0.5, float iou_thres = 0.45)
        : env_(ORT_LOGGING_LEVEL_WARNING, "Yolo11"), 
          session_options_(), 
          conf_threshold_(conf_thres), 
          iou_threshold_(iou_thres) 
    {
        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
        } catch (const std::exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            throw;
        }

        // 获取输入输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
        input_name_ = input_name_ptr.get();
        auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        input_h_ = input_shape[2];
        input_w_ = input_shape[3];

        // Output info
        auto output_name_ptr = session_->GetOutputNameAllocated(0, allocator);
        output_name_ = output_name_ptr.get();
    }

    std::vector<Detection> detect(cv::Mat& img, bool show_img) {
        // 1. 预处理 (Letterbox)
        cv::Mat input_img;
        float ratio = std::min((float)input_w_ / img.cols, (float)input_h_ / img.rows);
        int new_w = std::round(img.cols * ratio);
        int new_h = std::round(img.rows * ratio);
        int dw = (input_w_ - new_w) / 2;
        int dh = (input_h_ - new_h) / 2;

        cv::resize(img, input_img, cv::Size(new_w, new_h));
        cv::copyMakeBorder(input_img, input_img, dh, input_h_ - new_h - dh, dw, input_w_ - new_w - dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        // BlobFromImage (HWC -> CHW, Normalize)
        cv::Mat blob;
        cv::dnn::blobFromImage(input_img, blob, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), true, false);

        // 2. 推理
        std::vector<int64_t> input_dims = {1, 3, input_h_, input_w_};
        size_t input_tensor_size = 1 * 3 * input_h_ * input_w_;
        std::vector<float> input_tensor_values(input_tensor_size);
        input_tensor_values.assign((float*)blob.datastart, (float*)blob.dataend);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_dims.data(), input_dims.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // 3. 后处理 (YOLOv11/v8 Output shape: [1, 4 + nc, 8400])
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int rows = output_shape[2];       // 8400 anchors

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // 转置处理：OpenCV NMS 需要 [rows, dimensions] 格式，但 raw 是 [dimensions, rows]
        // 遍历所有 anchor
        for (int i = 0; i < rows; ++i) {            
            // 找出最大分数的类别
            float max_score = -1.0f;
            int class_id = -1;
            
            // 跳过前4个坐标 (x, y, w, h)
            for (int c = 0; c < 80; ++c) {
                // 索引计算：data[4 + c][i] -> 在展平数组中 stride 是 rows
                // raw output layout: [batch, channel, anchor]
                // value at (0, channel_idx, anchor_idx) = data[channel_idx * rows + anchor_idx]
                float score = output_data[(4 + c) * rows + i];
                if (score > max_score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score >= conf_threshold_) {
                // 读取坐标
                float cx = output_data[0 * rows + i];
                float cy = output_data[1 * rows + i];
                float w  = output_data[2 * rows + i];
                float h  = output_data[3 * rows + i];

                // 还原到原图尺寸
                float x = (cx - w / 2 - dw) / ratio;
                float y = (cy - h / 2 - dh) / ratio;
                float w_orig = w / ratio;
                float h_orig = h / ratio;

                boxes.push_back(cv::Rect(x, y, w_orig, h_orig));
                confidences.push_back(max_score);
                class_ids.push_back(class_id);
            }
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_threshold_, iou_threshold_, indices);

        std::vector<Detection> results;
        for (int idx : indices) {
            Detection det;
            det.class_id = class_ids[idx];
            det.confidence = confidences[idx];
            det.box = boxes[idx];
            det.center = cv::Point(det.box.x + det.box.width / 2, det.box.y + det.box.height / 2);
            results.push_back(det);
            
            if (show_img) {
                cv::rectangle(img, det.box, cv::Scalar(0, 255, 0), 2);
                std::string label = COCO_CLASSES[det.class_id] + " " + cv::format("%.2f", det.confidence);
                cv::putText(img, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }
        return results;
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    int64_t input_w_, input_h_;
    float conf_threshold_;
    float iou_threshold_;
};


// ================= ROS2 Node =================
class YoloDetectNode : public rclcpp::Node {
public:
    YoloDetectNode() : Node("yolo_detect_node") {
        // 1. 声明参数
        this->declare_parameter("camera.fx", 905.5593);
        this->declare_parameter("camera.fy", 905.5208);
        this->declare_parameter("camera.cx", 663.4498);
        this->declare_parameter("camera.cy", 366.7621);
        this->declare_parameter("conf_thres", 0.50);
        this->declare_parameter("show_image", true);
        this->declare_parameter("model_filename", "yolo11n.onnx"); 

        // 2. 获取参数
        fx_ = this->get_parameter("camera.fx").as_double();
        fy_ = this->get_parameter("camera.fy").as_double();
        cx_ = this->get_parameter("camera.cx").as_double();
        cy_ = this->get_parameter("camera.cy").as_double();
        conf_thres_ = this->get_parameter("conf_thres").as_double();
        show_image_ = this->get_parameter("show_image").as_bool();
        
        std::string model_filename = this->get_parameter("model_filename").as_string();
        std::string final_model_path;

        // 3. 智能路径处理逻辑
        namespace fs = std::filesystem;
        fs::path p(model_filename);

        if (p.is_absolute()) {
            // 如果用户传的是绝对路径（例如 /tmp/test.onnx），直接使用
            final_model_path = model_filename;
        } else {
            // 如果是相对路径，则去 share/detect11/models/ 下寻找
            try {
                std::string package_share_directory = ament_index_cpp::get_package_share_directory("detect11");
                // 拼接路径: share/detect11/models/yolo11n.onnx
                fs::path share_path(package_share_directory);
                fs::path full_path = share_path / "models" / model_filename;
                final_model_path = full_path.string();
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Can't find package 'detect11': %s", e.what());
                return;
            }
        }

        // 检查文件是否存在
        if (!fs::exists(final_model_path)) {
            RCLCPP_ERROR(this->get_logger(), "Model file does not exist: %s", final_model_path.c_str());
            return; // 或者抛出异常
        }

        RCLCPP_INFO(this->get_logger(), "Loading Model from: %s", final_model_path.c_str());
        
        try {
            detector_ = std::make_unique<Yolo11Detector>(final_model_path, conf_thres_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error initializing YOLO detector: %s", e.what());
            return;
        }

        // 3. 通信初始化
        // 使用 best_effort 或 sensor_data QoS
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(qos_profile), qos_profile);

        color_sub_.subscribe(this, "/camera/realsense_d435i/color/image_raw", qos.get_rmw_qos_profile());
        depth_sub_.subscribe(this, "/camera/realsense_d435i/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        // 同步器设置: Queue size 10, slop 0.1s
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), color_sub_, depth_sub_);
        sync_->registerCallback(&YoloDetectNode::sync_callback, this);
        sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.1));

        publisher_ = this->create_publisher<object3d_msgs::msg::Object3DArray>("target_points_array", qos);

        // 参数回调
        param_callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&YoloDetectNode::parametersCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "YoloDetectNode initialized.");
    }

private:
    // 参数回调
    rcl_interfaces::msg::SetParametersResult parametersCallback(
        const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param : parameters) {
            if (param.get_name() == "show_image" && param.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) {
                show_image_ = param.as_bool();
                RCLCPP_INFO(this->get_logger(), "Parameter 'show_image' updated to: %s", show_image_ ? "true" : "false");
            }
        }
        return result;
    }

    // 深度获取 (中值滤波)
    float get_robust_depth(const cv::Mat& depth_img, int cx, int cy) {
        if (cx < 0 || cx >= depth_img.cols || cy < 0 || cy >= depth_img.rows) return -1.0;

        int x_min = std::max(0, cx - 2);
        int x_max = std::min(depth_img.cols, cx + 3);
        int y_min = std::max(0, cy - 2);
        int y_max = std::min(depth_img.rows, cy + 3);

        std::vector<unsigned short> valid_depths;
        valid_depths.reserve(25);

        for (int y = y_min; y < y_max; ++y) {
            for (int x = x_min; x < x_max; ++x) {
                unsigned short d = depth_img.at<unsigned short>(y, x);
                if (d > 0) valid_depths.push_back(d);
            }
        }

        if (valid_depths.empty()) return -1.0;

        // 计算中值
        size_t n = valid_depths.size() / 2;
        std::nth_element(valid_depths.begin(), valid_depths.begin() + n, valid_depths.end());
        return (float)valid_depths[n];
    }

    void sync_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
        // FPS 计算
        auto now = std::chrono::steady_clock::now();
        frame_count_++;
        double elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time_).count();
        if (elapsed >= 1.0) {
            fps_ = frame_count_ / elapsed;
            frame_count_ = 0;
            start_time_ = now;
        }

        cv::Mat color_img, depth_img;
        try {
            color_img = cv_bridge::toCvShare(msg_color, "bgr8")->image;
            depth_img = cv_bridge::toCvShare(msg_depth, "16UC1")->image;
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // 绘制 FPS
        if (show_image_) {
            cv::putText(color_img, "FPS: " + std::to_string((int)fps_), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        } else {
             // 简单的日志限流
             static auto last_log = std::chrono::steady_clock::now();
             if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log).count() > 5) {
                 RCLCPP_INFO(this->get_logger(), "FPS: %.2f", fps_);
                 last_log = now;
             }
        }

        // 推理
        auto detections = detector_->detect(color_img, show_image_);

        if (show_image_) {
            const std::string win_name = "YOLOv11 Detection";
            
            // 1. 创建窗口（允许调整大小）
            cv::namedWindow(win_name, cv::WINDOW_NORMAL);
            
            // 2. 设置全屏属性
            cv::setWindowProperty(win_name, cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            
            // 3. 显示
            cv::imshow(win_name, color_img);
            cv::waitKey(1);
        } else {
            // 如果不需要显示，且窗口存在，则关闭它
            try {
                // 检查窗口是否还开着，开着就关掉
                if (cv::getWindowProperty("YOLOv11 Detection", cv::WND_PROP_VISIBLE) >= 0) {
                    cv::destroyWindow("YOLOv11 Detection");
                    cv::waitKey(1); // 刷新事件
                }
            } catch (...) {
                // 忽略异常
            }
        }

        // 如果没有订阅者，跳过 3D 计算
        if (publisher_->get_subscription_count() == 0 || detections.empty()) return;

        object3d_msgs::msg::Object3DArray array_msg;
        array_msg.header = msg_color->header;

        for (const auto& det : detections) {
            float d = get_robust_depth(depth_img, det.center.x, det.center.y);
            
            if (d <= 0) continue;

            float Z = d / 1000.0f; // mm to meters
            float X = (det.center.x - cx_) * Z / fx_;
            float Y = (det.center.y - cy_) * Z / fy_;

            object3d_msgs::msg::Object3D obj;
            obj.point.x = X;
            obj.point.y = Y;
            obj.point.z = Z;
            
            // 像素宽 * Z / fx = 物理宽
            obj.width_m = (det.box.width * Z) / fx_;
            obj.height_m = (det.box.height * Z) / fy_;
            
            obj.class_name = COCO_CLASSES[det.class_id];
            obj.score = det.confidence;

            array_msg.objects.push_back(obj);
        }

        if (!array_msg.objects.empty()) {
            publisher_->publish(array_msg);
        }
    }

    // 成员变量
    double fx_, fy_, cx_, cy_;
    double conf_thres_;
    bool show_image_;
    
    std::unique_ptr<Yolo11Detector> detector_;
    
    message_filters::Subscriber<sensor_msgs::msg::Image> color_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
    
    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr publisher_;
    OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;

    // FPS 相关
    std::chrono::steady_clock::time_point start_time_ = std::chrono::steady_clock::now();
    int frame_count_ = 0;
    double fps_ = 0.0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloDetectNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}