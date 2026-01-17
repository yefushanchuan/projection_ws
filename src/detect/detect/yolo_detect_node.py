import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
from ament_index_python.packages import get_package_share_directory
from detect.bpu_detect_hobot_dnn import BPU_Detect
from rcl_interfaces.msg import SetParametersResult 
from object3d_msgs.msg import Object3D, Object3DArray
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import message_filters

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')

        # 1. 声明参数
        self.declare_parameter('camera.fx', 905.5593)
        self.declare_parameter('camera.fy', 905.5208)
        self.declare_parameter('camera.cx', 663.4498)
        self.declare_parameter('camera.cy', 366.7621)
        self.declare_parameter('conf_thres', 0.50)
        self.declare_parameter('show_image', True)
        # 默认模型文件名
        self.declare_parameter('model_filename', 'yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin')
        
        # 2. 获取参数值
        self.camera_fx = self.get_parameter('camera.fx').get_parameter_value().double_value
        self.camera_fy = self.get_parameter('camera.fy').get_parameter_value().double_value
        self.camera_cx = self.get_parameter('camera.cx').get_parameter_value().double_value
        self.camera_cy = self.get_parameter('camera.cy').get_parameter_value().double_value
        self.show_image_flag = self.get_parameter('show_image').get_parameter_value().bool_value
        conf_val = self.get_parameter('conf_thres').get_parameter_value().double_value
        model_filename = self.get_parameter('model_filename').get_parameter_value().string_value

        # 3. 初始化通信
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/realsense_d435i/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/realsense_d435i/aligned_depth_to_color/image_raw', qos_profile=qos_profile_sensor_data)

        # ApproximateTimeSynchronizer: 
        # 参数2: queue_size=50 (缓冲队列长度)
        # 参数3: slop=0.2 (允许 200ms 的时间误差，RealSense通常在30ms以内)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 50, 0.2)
        self.ts.registerCallback(self.sync_callback)

        self.publisher_ = self.create_publisher(Object3DArray, 'target_points_array', qos_profile_sensor_data)
        
        self.bridge = CvBridge()
        self.depth_image = None

        self.latest_color_msg = None
        self.latest_depth_msg = None

        # FPS 计算变量
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.fps_log_counter = 0

        # COCO 类别名称 (80类)
        self.labelname = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
            "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
            "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]

        # 4. 模型路径处理
        if os.path.isabs(model_filename):
            model_path = model_filename
            if not os.path.exists(model_path):
                self.get_logger().error(f"Error: Model file not found at {model_path}")
        else:
            try:
                pkg_path = get_package_share_directory('detect')
                model_path = os.path.join(pkg_path, 'models', model_filename)
            except Exception as e:
                self.get_logger().error(f"Error finding package path: {e}")
                model_path = ""

        self.get_logger().info(f"Loading Model: {model_path}")

        # 5. 初始化推理引擎
        self.detector = BPU_Detect(
            model_path = model_path,
            labelnames = self.labelname,
            conf = conf_val,
            is_save = False
        )

        # 6. 参数回调注册
        self.add_on_set_parameters_callback(self.parameter_show_image_flag_callback)

    def parameter_show_image_flag_callback(self, params):
        """处理来自 Qt 界面的参数修改请求"""
        for param in params:
            if param.name == 'show_image' and param.type_ == param.Type.BOOL:
                self.show_image_flag = param.value
                self.get_logger().info(f"Qt Signal: show_image -> {self.show_image_flag}")
        
        return SetParametersResult(successful=True)

    def listener_callback(self, msg):
        self.latest_color_msg = msg

    def depth_callback(self, msg):
        self.latest_depth_msg = msg

    def get_robust_depth(self, depth_img, cx, cy):
        h, w = depth_img.shape
        cx, cy = int(cx), int(cy)
        
        # 1. 定义 5x5 窗口边界 (防止越界)
        x_min = max(0, cx - 2)
        x_max = min(w, cx + 3)  # slice 前闭后开
        y_min = max(0, cy - 2)
        y_max = min(h, cy + 3)
        
        # 2. 切片提取 ROI
        roi = depth_img[y_min:y_max, x_min:x_max]
        
        # 3. 剔除 0 值 (无效深度)
        valid_pixels = roi[roi > 0]
        
        # 4. 如果没有有效值，返回 -1
        if len(valid_pixels) == 0:
            return -1
                
        # 5. 计算中值 (Median)
        median_depth = np.median(valid_pixels)
        
        return float(median_depth)

    def sync_callback(self, msg_color, msg_depth):
        # --- 1. 转图像 ---
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(msg_depth, '16UC1')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        show_img = self.show_image_flag

        # --- 2. FPS 计算 ---
        self.frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = curr_time

        if show_img:
            cv2.putText(cv_image, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # --- 3. 推理 (BPU) ---
        # 注意：这里会阻塞回调，如果推理太慢且 fps 很高，message_filters 会自动丢弃处理不过来的帧
        self.detector.detect(cv_image, show_img=show_img)

        # --- 4. 没订阅者就别算 3D ---
        if self.publisher_.get_subscription_count() == 0:
            return

        if not hasattr(self.detector, 'centers') or len(self.detector.centers) == 0:
            return

        # --- 5. 计算 3D 并发布 ---
        array_msg = Object3DArray()
        # 重要：使用同步后，header 应该用谁的？通常 RGB 和 Depth 时间戳近似，用 RGB 的即可
        array_msg.header = msg_color.header

        fx, fy = self.camera_fx, self.camera_fy
        cx_, cy_ = self.camera_cx, self.camera_cy

        for i, (cx, cy) in enumerate(self.detector.centers):
            # 调用改进后的深度获取函数
            d = self.get_robust_depth(depth_image, cx, cy)
            
            if d <= 0:
                continue

            Z = d / 1000.0
            X = (cx - cx_) * Z / fx
            Y = (cy - cy_) * Z / fy

            bbox = self.detector.bboxes[i]
            w_pixel = bbox[2] - bbox[0]
            h_pixel = bbox[3] - bbox[1]

            obj = Object3D()
            obj.point.x = X
            obj.point.y = Y
            obj.point.z = Z
            obj.width_m = (w_pixel * Z) / fx
            obj.height_m = (h_pixel * Z) / fy

            try:
                cid = int(self.detector.ids[i])
                obj.class_name = self.labelname[cid]
                obj.score = float(self.detector.scores[i])
            except:
                obj.class_name = "unknown"
                obj.score = 0.0

            array_msg.objects.append(obj)

        if array_msg.objects:
            self.publisher_.publish(array_msg)
        
def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()