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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')

        # QoS 设置
        latest_frame_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,  # 配合你现在的 Reliable 相机
            history=HistoryPolicy.KEEP_LAST,
            depth=1                                  # <--- 关键！只要最新的一张
        )

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
        self.create_subscription(Image, '/camera/realsense_d435i/color/image_raw', self.listener_callback, latest_frame_qos)
        self.create_subscription(Image, '/camera/realsense_d435i/aligned_depth_to_color/image_raw', self.depth_callback, latest_frame_qos)
        self.publisher_ = self.create_publisher(Object3DArray, 'target_points_array', latest_frame_qos)
        
        self.bridge = CvBridge()
        self.depth_image = None
        
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
        try:
            # 1. 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            show_img = self.show_image_flag # 获取当前最新的开关状态

            # 2. FPS 计算
            self.frame_count += 1
            curr_time = time.time()
            elapsed_time = curr_time - self.start_time

            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = curr_time
                self.fps_log_counter += 1
                
                # 优化日志：仅当不显示界面时，每5秒打印一次心跳包
                if not show_img and (self.fps_log_counter % 5 == 0):
                    self.get_logger().info(f"Node Running - FPS: {self.fps:.2f}")

            # 3. 绘制 FPS 到图像上
            if show_img:
                cv2.putText(cv_image, f"FPS: {self.fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # 4. 执行推理 (内部会处理 show_img 逻辑，包括销毁窗口)
            self.detector.detect(cv_image, show_img=show_img)

            # 5. 结合深度信息发布 3D 坐标
            if self.publisher_.get_subscription_count() > 0 and hasattr(self.detector, 'centers') and len(self.detector.centers) > 0:
                # 只有当深度图准备好时才计算
                if self.depth_image is None:
                    # 避免在启动初期疯狂打印 warning，可以使用 debug
                    self.get_logger().debug("Waiting for depth image...") 
                    return

                array_msg = Object3DArray()
                array_msg.header = msg.header

                # 相机内参 (从参数服务器获取最新的)
                fx, fy = self.camera_fx, self.camera_fy
                cx_, cy_ = self.camera_cx, self.camera_cy

                for i, (cx, cy) in enumerate(self.detector.centers):
                    # 获取深度 Z
                    depth_value = self.get_depth_value(int(cx), int(cy))
                    
                    # 过滤无效深度
                    if depth_value <= 0:
                        continue

                    # 坐标转换: 像素坐标 (u,v,d) -> 相机坐标 (X,Y,Z)
                    Z = float(depth_value) / 1000.0 # 毫米转米
                    X = (float(cx) - cx_) * Z / fx
                    Y = (float(cy) - cy_) * Z / fy

                    # === 计算物理宽高 ===
                    bbox = self.detector.bboxes[i] # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bbox
                    w_pixel = x2 - x1
                    h_pixel = y2 - y1

                    # 物理长度 = (像素长度 * 距离Z) / 焦距
                    # 宽度对应 fx，高度对应 fy
                    width_m = (w_pixel * Z) / self.camera_fx
                    height_m = (h_pixel * Z) / self.camera_fy
                    # 构建消息
                    obj_msg = Object3D()
                    obj_msg.point.x = X
                    obj_msg.point.y = Y
                    obj_msg.point.z = Z
                    obj_msg.width_m = width_m
                    obj_msg.height_m = height_m

                    # 获取类别和置信度
                    try:
                        class_id = int(self.detector.ids[i])
                        obj_msg.class_name = self.labelname[class_id]
                        obj_msg.score = float(self.detector.scores[i])
                    except (IndexError, AttributeError):
                        obj_msg.class_name = "unknown"
                        obj_msg.score = 0.0

                    array_msg.objects.append(obj_msg)

                    # Debug 日志：不会刷屏，除非你开启 debug 级别
                    self.get_logger().debug(f"Det: {obj_msg.class_name} at ({X:.2f}, {Y:.2f}, {Z:.2f})")
                    
                if len(array_msg.objects) > 0:
                    self.publisher_.publish(array_msg)

        except Exception as e:  
            self.get_logger().error(f"Inference Loop Error: {e}")

    def depth_callback(self, msg):
        try:
            # 16UC1 是 Realsense 标准深度格式 (单位 mm)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except Exception as e:
            self.get_logger().error(f"Depth convert error: {e}")

    def get_depth_value(self, cx, cy):
        # 增加健壮性检查：防止坐标越界导致崩溃
        if self.depth_image is None:
            return -1
            
        h, w = self.depth_image.shape
        if 0 <= cx < w and 0 <= cy < h:
            val = self.depth_image[cy, cx]
            return val if val > 0 else -1
        else:
            # 只有越界时才打印 debug
            self.get_logger().debug(f"Coord out of bounds: ({cx}, {cy}) vs Img({w}x{h})")
            return -1
        
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