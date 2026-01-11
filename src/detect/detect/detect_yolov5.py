import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
from ament_index_python.packages import get_package_share_directory
#from detect.bpu_infer_custom import BPU_Detect
from detect.bpu_infer_hobot import BPU_Detect
from object3d_msgs.msg import Object3D, Object3DArray
from rcl_interfaces.msg import SetParametersResult 

class YoloDetectNode(Node):
    def __init__(self):
        super().__init__('yolo_detect_node')

        self.declare_parameter('camera.fx', 905.5593)
        self.declare_parameter('camera.fy', 905.5208)
        self.declare_parameter('camera.cx', 663.4498)
        self.declare_parameter('camera.cy', 366.7621)

        self.declare_parameter('conf_thres', 0.35)
        self.declare_parameter('show_image', True)
        self.declare_parameter('model_filename', 'yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin')
        
        self.camera_fx = self.get_parameter('camera.fx').get_parameter_value().double_value
        self.camera_fy = self.get_parameter('camera.fy').get_parameter_value().double_value
        self.camera_cx = self.get_parameter('camera.cx').get_parameter_value().double_value
        self.camera_cy = self.get_parameter('camera.cy').get_parameter_value().double_value
        self.show_image_flag = self.get_parameter('show_image').get_parameter_value().bool_value

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10
        )
        self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',  # 订阅深度图像
            self.depth_callback,
            10
        )
        self.publisher_ = self.create_publisher(
            Object3DArray,
            'target_points_array',
            10
        )
        self.bridge = CvBridge()
        self.depth_image = None
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.fps_log_counter = 0

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

        conf_val = self.get_parameter('conf_thres').get_parameter_value().double_value
        
        model_filename = self.get_parameter('model_filename').get_parameter_value().string_value
        
        #  !!! 智能路径判断逻辑 !!!
        if os.path.isabs(model_filename):
            # 情况 A: 用户在 Qt 里选了文件 (例如: /home/sunrise/my_models/best.bin)
            model_path = model_filename
            if not os.path.exists(model_path):
                self.get_logger().error(f"自定义路径模型不存在: {model_path}")
                # 可以在这里做容错处理，比如回退到默认模型
        else:
            # 情况 B: 用户没改 Qt 输入框，传过来的是默认文件名
            try:
                pkg_path = get_package_share_directory('detect')
                model_path = os.path.join(pkg_path, 'models', model_filename)
            except Exception as e:
                self.get_logger().error(f"查找功能包路径失败: {e}")
                model_path = ""

        self.get_logger().info(f"正在加载模型: {model_path}")

        self.detector = BPU_Detect(
            model_path = model_path,
            labelnames = self.labelname,
            conf = conf_val,
#            mode = True,
            is_save = False
            )

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'show_image':
                if param.type_ == param.Type.BOOL:
                    self.show_image_flag = param.value
                    # self.get_logger().info(f"Updated show_image to: {self.show_image_flag}")
        return SetParametersResult(successful=True)

    def listener_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            show_img = self.show_image_flag

            self.frame_count += 1
            curr_time = time.time()
            elapsed_time = curr_time - self.start_time

            if elapsed_time >= 1.0:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = curr_time
                
                self.fps_log_counter += 1
                if not show_img and (self.fps_log_counter % 5 == 0):
                    self.get_logger().info(f"Current FPS: {self.fps:.2f}")

            if show_img:
                cv2.putText(cv_image, f"FPS: {self.fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            self.detector.detect(cv_image, show_img=show_img)

            array_msg = Object3DArray()
            array_msg.header = msg.header

            fx = self.camera_fx
            fy = self.camera_fy
            cx_ = self.camera_cx
            cy_ = self.camera_cy

            if hasattr(self.detector, 'centers') and len(self.detector.centers) > 0 and self.depth_image is not None:
                for i, (cx, cy) in enumerate(self.detector.centers):
                    depth_value = self.get_depth_value(int(cx), int(cy))
                    if depth_value <= 0:
                        continue

                    target_depth_msg = Object3D()

                    Z = float(depth_value) / 1000.0
                    X = (float(cx) - cx_) * Z / fx
                    Y = (float(cy) - cy_) * Z / fy

                    target_depth_msg.point.x = X
                    target_depth_msg.point.y = Y
                    target_depth_msg.point.z = Z

                    try:
                        class_id = int(self.detector.ids[i])
                        score_val = float(self.detector.scores[i])
                        
                        target_depth_msg.class_name = self.labelname[class_id]
                        target_depth_msg.score = score_val
                    except IndexError:
                        target_depth_msg.class_name = "unknown"
                        target_depth_msg.score = 0.0

                    array_msg.objects.append(target_depth_msg)

                    self.get_logger().debug(
                        f"Add Point: {target_depth_msg.class_name}, ({X:.2f}, {Y:.2f}, {Z:.2f})"
                    )
                    
                if len(array_msg.objects) > 0:
                    self.publisher_.publish(array_msg)

        except Exception as e:  
            self.get_logger().error(f"Failed to convert image: {e}")

    def depth_callback(self, msg):
        try:
            # 将 ROS 深度图像消息转换为 OpenCV 图像
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')  # 16UC1是深度图像的编码格式
        except Exception as e:
            self.get_logger().error(f"Failed to convert depth image: {e}")

    def get_depth_value(self, cx, cy):
#        cx = int(cx * 848 / 1280)
#        cy = int(cy * 480 / 720)
        if 0 <= cx < self.depth_image.shape[1] and 0 <= cy < self.depth_image.shape[0]:
            depth_value = self.depth_image[cy, cx]
            if depth_value > 0:
                return depth_value
            else:
                return -1  # 0深度也返回-1
        else:
            self.get_logger().debug(f"Invalid coordinates ({cx}, {cy}) for depth image")
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
