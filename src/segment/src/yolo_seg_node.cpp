#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "dnn_node/dnn_node.h"
#include "dnn_node/util/image_proc.h"
#include <chrono>
#include <fstream>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "object3d_msgs/msg/object3_d.hpp"
#include "object3d_msgs/msg/object3_d_array.hpp"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "segment/yolo_seg_common.h"
#include "segment/bpu_seg_dnn.h"

using hobot::dnn_node::DNNInput;
using hobot::dnn_node::DnnNodeOutput;

class YoloSegNode : public hobot::dnn_node::DnnNode {
public:
    YoloSegNode(const std::string& node_name, const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : hobot::dnn_node::DnnNode(node_name, options) {
        // 1. 参数声明与获取
        // 默认值只写文件名即可，程序会自动去 share/segment/models/ 下找
        this->declare_parameter("model_filename", "yolo11x_seg_bayese_640x640_nv12.bin");
        this->declare_parameter("show_image", true);
        this->declare_parameter("conf_thres", 0.50);
        this->declare_parameter("nms_thres", 0.45);

        // 相机内参 (默认值来自你的 Detect 代码)
        this->declare_parameter("camera.fx", 905.5593);
        this->declare_parameter("camera.fy", 905.5208);
        this->declare_parameter("camera.cx", 663.4498);
        this->declare_parameter("camera.cy", 366.7621);

        this->get_parameter("camera.fx", fx_);
        this->get_parameter("camera.fy", fy_);
        this->get_parameter("camera.cx", cx_);
        this->get_parameter("camera.cy", cy_);

        // 2. 智能路径解析 (仿 Python 逻辑)
        std::string model_param = this->get_parameter("model_filename").as_string();
        
        // 判断是否为绝对路径 (以 / 开头)
        if (model_param.front() == '/') {
            resolved_model_path_ = model_param;
        } else {
            // 如果是相对路径/文件名，则拼接 share 目录
            try {
                resolved_model_path_ = ament_index_cpp::get_package_share_directory("segment") + "/models/" + model_param;
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Can not find package 'segment': %s", e.what());
                return;
            }
        }

        // 检查文件是否存在
        std::ifstream f(resolved_model_path_.c_str());
        if (!f.good()) {
            RCLCPP_ERROR(this->get_logger(), "Model file not found: %s", resolved_model_path_.c_str());
            // 这里不 return，让 Init() 去报错，或者你可以选择直接 throw
        } else {
            RCLCPP_INFO(this->get_logger(), "Loading Model: %s", resolved_model_path_.c_str());
        }

        // 3. 初始化 DNN (Init 会调用 SetNodePara)
        if (Init() != 0) {
            RCLCPP_ERROR(this->get_logger(), "Init failed!");
            return;
        }

        if (GetModelInputSize(0, model_input_w_, model_input_h_) < 0) {
             RCLCPP_ERROR(this->get_logger(), "Get model input size failed!");
        }

        segmenter_ = std::make_shared<BPU_Segment>();

        this->get_parameter("show_image", show_img_);
        this->get_parameter("conf_thres", conf_thres_);
        this->get_parameter("nms_thres", nms_thres_);
        
        // 将成员变量的值同步给算法引擎
        segmenter_->config_.conf_thres = (float)conf_thres_;
        segmenter_->config_.nms_thres = (float)nms_thres_;

        callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&YoloSegNode::parameter_callback, this, std::placeholders::_1));

        // 4. QoS 设置
        auto qos = rclcpp::SensorDataQoS().keep_last(1);
        
        sub_color_filter_.subscribe(this, "/camera/realsense_d435i/color/image_raw", qos.get_rmw_qos_profile());
        sub_depth_filter_.subscribe(this, "/camera/realsense_d435i/aligned_depth_to_color/image_raw", qos.get_rmw_qos_profile());

        // 初始化同步器
        // Queue size = 10, Slop = 0.1s (允许100ms误差)
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), sub_color_filter_, sub_depth_filter_
        );
        
        // 注册同步回调
        sync_->registerCallback(std::bind(&YoloSegNode::SyncCallback, this, std::placeholders::_1, std::placeholders::_2));

        pub_ = this->create_publisher<object3d_msgs::msg::Object3DArray>("target_points_array", qos);

        last_calc_time_ = std::chrono::steady_clock::now();
        last_log_time_ = std::chrono::steady_clock::now();
    }

    ~YoloSegNode() override {
    cv::destroyAllWindows(); 
    }
        
    float GetRobustDepth(const cv::Mat& depth_img, int cx, int cy)
    {
        if (depth_img.empty()) return -1.0f;

        int h = depth_img.rows;
        int w = depth_img.cols;

        std::vector<uint16_t> valid_depths;
        valid_depths.reserve(25); // 5x5 = 25

        // 遍历 5x5 窗口
        for (int dy = -2; dy <= 2; ++dy) {
            for (int dx = -2; dx <= 2; ++dx) {
                int u = cx + dx;
                int v = cy + dy;

                // 边界检查
                if (u >= 0 && u < w && v >= 0 && v < h) {
                    uint16_t d = depth_img.at<uint16_t>(v, u);
                    // 剔除无效值 0
                    if (d > 0) {
                        valid_depths.push_back(d);
                    }
                }
            }
        }

        if (valid_depths.empty()) {
            return -1.0f;
        }

        // 中值滤波（nth_element，O(n)）
        size_t n = valid_depths.size() / 2;
        std::nth_element(
            valid_depths.begin(),
            valid_depths.begin() + n,
            valid_depths.end()
        );

        uint16_t median_val = valid_depths[n];

        return static_cast<float>(median_val) / 1000.0f; // mm -> m
    }

protected:
    int SetNodePara() override {
        if (!dnn_node_para_ptr_) return -1;
        // 使用我们在构造函数中解析好的绝对路径
        dnn_node_para_ptr_->model_file = resolved_model_path_;
        dnn_node_para_ptr_->task_num = 2; 
        return 0;
    }

    int PostProcess(const std::shared_ptr<DnnNodeOutput>& node_output) override {
        if (!node_output) return -1;

        auto seg_output = std::dynamic_pointer_cast<YoloSegOutput>(node_output);
        if (!seg_output || !seg_output->src_img || !seg_output->depth_img) return -1;

        UpdateFPS(show_img_);

        // 算法后处理
        std::vector<SegResult> results;
        segmenter_->PostProcess(node_output->output_tensors, model_input_h_, model_input_w_, results);

        PublishSegMessage(results, seg_output->msg_header, *(seg_output->depth_img));

        // 可视化
        if (show_img_) {
            cv::Mat draw_img = seg_output->src_img->clone();
            segmenter_->detect_result(draw_img, results, fps_, true);
        } else {
            cv::Mat dummy_img; 
            segmenter_->detect_result(dummy_img, results, fps_, false);
        }

        return 0;
    }

private:
    std::string resolved_model_path_; // 存储解析后的模型路径
    int model_input_w_;
    int model_input_h_;
    bool show_img_; 
    double conf_thres_;
    double nms_thres_;
    double fx_, fy_, cx_, cy_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> SyncPolicy;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_color_filter_;
    message_filters::Subscriber<sensor_msgs::msg::Image> sub_depth_filter_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<object3d_msgs::msg::Object3DArray>::SharedPtr pub_;
    OnSetParametersCallbackHandle::SharedPtr callback_handle_;
    std::shared_ptr<BPU_Segment> segmenter_;

    std::chrono::steady_clock::time_point last_calc_time_;
    std::chrono::steady_clock::time_point last_log_time_;
    int frame_count_ = 0;
    double fps_ = 0.0;

    void PublishSegMessage(const std::vector<SegResult>& results, 
                           const std::shared_ptr<std_msgs::msg::Header>& header,
                           const cv::Mat& depth_mat) {
        
        if (pub_->get_subscription_count() == 0) return;

        object3d_msgs::msg::Object3DArray msg;
        msg.header = *header;

        for (const auto& res : results) {
            int u = std::round(res.mic_center.x);
            int v = std::round(res.mic_center.y);

            float z_m = GetRobustDepth(depth_mat, u, v);
            
            // 如果 z_m 返回 -1 (无效) 或者太近/太远，过滤掉
            if (z_m <= 0.0f) continue; 

            float x_m = (static_cast<float>(u) - cx_) * z_m / fx_;
            float y_m = (static_cast<float>(v) - cy_) * z_m / fy_;

            float radius_m = (res.mic_radius * z_m) / fx_;
            
            object3d_msgs::msg::Object3D obj;
            obj.class_name = res.class_name;
            obj.score = res.score;
            obj.point.x = x_m;
            obj.point.y = y_m;
            obj.point.z = z_m;
            obj.width_m = radius_m * 2.0f;
            obj.height_m = radius_m * 2.0f;

            msg.objects.push_back(obj);
        }

        if (!msg.objects.empty()) {
            pub_->publish(msg);
        }
    }

    rcl_interfaces::msg::SetParametersResult parameter_callback(
        const std::vector<rclcpp::Parameter> &params) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";

        for (const auto &param : params) {
            if (param.get_name() == "conf_thres") {
                if (segmenter_) {
                    segmenter_->config_.conf_thres = (float)param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated conf_thres: %.2f", segmenter_->config_.conf_thres);
                }
            }
            else if (param.get_name() == "nms_thres") {
                if (segmenter_) {
                    segmenter_->config_.nms_thres = (float)param.as_double();
                    RCLCPP_INFO(this->get_logger(), "Updated nms_thres: %.2f", segmenter_->config_.nms_thres);
                }
            }
            else if (param.get_name() == "show_image") {
                show_img_ = param.as_bool();
                RCLCPP_INFO(this->get_logger(), "Updated show_image: %s", show_img_ ? "true" : "false");
            }
        }
        return result;
    }

    void UpdateFPS(bool show_img) {
        auto now = std::chrono::steady_clock::now();
        frame_count_++;
        auto calc_dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_calc_time_).count();
        
        if (calc_dur >= 1000) { 
            fps_ = frame_count_ * 1000.0 / calc_dur;
            frame_count_ = 0;
            last_calc_time_ = now;
        }

        if (!show_img) {
            auto log_dur = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_time_).count();
            if (log_dur >= 5000) { 
                RCLCPP_INFO(this->get_logger(), "FPS: %.2f", fps_);
                last_log_time_ = now;
            }
        }
    }

    void SyncCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg_color, 
                      const sensor_msgs::msg::Image::ConstSharedPtr& msg_depth) {
        // 1. 转换图像
        cv::Mat img_color, img_depth;
        try {
            img_color = cv_bridge::toCvShare(msg_color, "bgr8")->image.clone();
            // Realsense 深度图通常是 16UC1
            img_depth = cv_bridge::toCvShare(msg_depth, "16UC1")->image.clone();
            
            // 如果需要 resize，必须在这里同时对 color 和 depth 进行 resize
            // 但建议直接在 launch 里设置相机出 640x480，这里就不需要消耗 CPU 做 resize 了
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
            return;
        }

        // 2. 准备 DNN 输入 (NV12 转换)
        cv::Mat nv12_mat;
        segmenter_->PreProcess(img_color, model_input_w_, model_input_h_, nv12_mat);

        auto pyramid = hobot::dnn_node::ImageProc::GetNV12PyramidFromNV12Img(
            reinterpret_cast<const char*>(nv12_mat.data),
            model_input_h_, model_input_w_,
            model_input_h_, model_input_w_
        );

        auto inputs = std::vector<std::shared_ptr<DNNInput>>{pyramid};

        // 3. 构造输出结构体 (携带同步好的 Depth)
        auto output = std::make_shared<YoloSegOutput>();
        output->msg_header = std::make_shared<std_msgs::msg::Header>(msg_color->header);
        output->src_img = std::make_shared<cv::Mat>(img_color);
        output->depth_img = std::make_shared<cv::Mat>(img_depth);

        // 4. 执行推理
        // 注意：Run 可能会阻塞或者排队，这取决于底层 BPU 调度
        // 如果输入帧率过高，dnn_node 内部可能会丢帧
        Run(inputs, output, nullptr, false);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<YoloSegNode>("yolo_seg_node"));
    rclcpp::shutdown();
    return 0;
}