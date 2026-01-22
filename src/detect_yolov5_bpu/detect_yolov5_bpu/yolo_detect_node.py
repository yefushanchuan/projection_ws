import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import time
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import SetParametersResult 
from object3d_msgs.msg import Object3D, Object3DArray
from rclpy.qos import qos_profile_sensor_data
import message_filters
from pathlib import Path

# 导入我们的新模块
from detect_yolov5_bpu.bpu_detect_hobot_dnn import BPU_Detect
import detect_yolov5_bpu.utils as utils

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')

        # 1. 参数声明 (保持不变)
        self.declare_parameter('camera.fx', 905.5593)
        self.declare_parameter('camera.fy', 905.5208)
        self.declare_parameter('camera.cx', 663.4498)
        self.declare_parameter('camera.cy', 366.7621)
        self.declare_parameter('conf_thres', 0.50)
        self.declare_parameter('nms_thres', 0.45)
        self.declare_parameter('show_image', True)
        self.declare_parameter('model_filename', 'yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin')
        
        # 获取参数
        self.fx = self.get_parameter('camera.fx').value
        self.fy = self.get_parameter('camera.fy').value
        self.cx = self.get_parameter('camera.cx').value
        self.cy = self.get_parameter('camera.cy').value
        self.show_image_flag = self.get_parameter('show_image').value
        conf_val = self.get_parameter('conf_thres').value
        nms_val = self.get_parameter('nms_thres').value
        model_filename = self.get_parameter('model_filename').value

        self.win_state = {'created': False} 
        if self.show_image_flag:
            dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(dummy_img, "Loading Model...", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # 立即调用一次显示，创建窗口
            utils.show_window("Detection Result", dummy_img, [], self.win_state)
            # 强制刷新一下事件队列
            cv2.waitKey(1) 

        # 2. 初始化通信
        self.color_sub = message_filters.Subscriber(self, Image, '/camera/realsense_d435i/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(self, Image, '/camera/realsense_d435i/aligned_depth_to_color/image_raw', qos_profile=qos_profile_sensor_data)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.sync_callback)
        
        self.publisher_ = self.create_publisher(Object3DArray, 'target_points_array', qos_profile_sensor_data)
        self.bridge = CvBridge()

        # 3. 变量初始化
        self.frame_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.fps = 0.0
        # 窗口状态字典 (用于引用传递)
        self.win_state = {'created': False} 

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

        # 4. 加载模型
        if os.path.isabs(model_filename):
            model_path = model_filename
        else:
            try:
                # 1. 获取包路径并转为 Path 对象
                p = Path(get_package_share_directory('detect_yolov5_bpu'))
                
                # 2. 向上查找直到找到 'install' 目录
                while p.name != 'install' and p != p.parent:
                    p = p.parent
                
                # 3. 拼接 workspace/models 路径
                if p.name == 'install':
                    model_path = str(p.parent / 'models' / model_filename)
                else:
                    self.get_logger().error("Failed to locate 'install' directory.")
                    return
            except Exception as e:
                self.get_logger().error(f"Path error: {e}")
                return

        # 检查与加载
        if not os.path.exists(model_path):
            self.get_logger().error(f"[Error] Model not found: {model_path}")
            return

        self.get_logger().info(f"Loading model: {model_path}")
        self.detector = BPU_Detect(model_path, self.labelname, conf=conf_val, iou=nms_val, is_save=False)
        self.add_on_set_parameters_callback(self.parameter_callback)
        self.get_logger().info("YOLOv5 BPU Node Initialized.")

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'show_image':
                self.show_image_flag = param.value
        return SetParametersResult(successful=True)

    def sync_callback(self, msg_color, msg_depth):
        # 1. FPS 计算
        self.frame_count += 1
        curr_time = time.time()
        if curr_time - self.start_time >= 1.0:
            self.fps = self.frame_count / (curr_time - self.start_time)
            self.frame_count = 0
            self.start_time = curr_time

        # 2. 图像转换
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg_color, 'bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(msg_depth, '16UC1')
        except Exception as e:
            return

        # 3. 推理 (返回 UnifiedResult 列表)
        results = self.detector.detect(cv_image)

        # 4. 3D 计算与发布
        if self.publisher_.get_subscription_count() > 0:
            array_msg = Object3DArray()
            array_msg.header = msg_color.header

            for res in results:
                # 使用 utils 提取深度
                d_m = utils.get_robust_depth(depth_image, res.center[0], res.center[1])
                
                if d_m <= 0: continue

                # 投影计算
                X = (res.center[0] - self.cx) * d_m / self.fx
                Y = (res.center[1] - self.cy) * d_m / self.fy
                
                obj = Object3D()
                obj.point.x = X
                obj.point.y = Y
                obj.point.z = d_m
                obj.width_m = (res.box[2] * d_m) / self.fx
                obj.height_m = (res.box[3] * d_m) / self.fy
                obj.class_name = res.class_name
                obj.score = res.score
                
                array_msg.objects.append(obj)

            if array_msg.objects:
                self.publisher_.publish(array_msg)

        # 5. 可视化 (使用 utils)
        if self.show_image_flag:
            utils.show_window("Detection Result", cv_image, results, self.win_state, self.fps)
        else:
            # 处理窗口关闭
            if self.win_state['created']:
                cv2.destroyWindow("Detection Result")
                self.win_state['created'] = False
                cv2.waitKey(1)
            
            # 打印日志代替显示
            if curr_time - self.last_log_time >= 5.0:
                self.get_logger().info(f"FPS: {self.fps:.2f}, Objects: {len(results)}")
                self.last_log_time = curr_time

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(YoloDetectNode())
    rclpy.shutdown()