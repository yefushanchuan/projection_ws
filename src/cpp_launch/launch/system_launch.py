import os
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction, LogInfo, ExecuteProcess, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # ==========================================
    # 0. 动态获取工作空间根目录
    # ==========================================
    current_path = Path(__file__).resolve()
    
    # 向上遍历，直到找到 'install' 目录 或者 'src' 目录
    # 这样写的好处：无论你是 colcon build 还是 colcon build --symlink-install 都能找到
    while current_path.name != 'install' and current_path.name != 'src':
        if current_path == current_path.parent:
            # 这是一个保险措施：如果回退到系统根目录 '/' 还没找到，就停止
            print("\033[91m[Error] Could not find workspace root (install/src)!\033[0m")
            break
        current_path = current_path.parent

    # current_path 此时指向 'install' 或 'src' 文件夹
    # 它的父级就是 工作空间根目录 (workspace)
    workspace_root = current_path.parent
    
    # 拼接 models 目录
    workspace_models_path = workspace_root / 'models'
    
    # 拼接具体的模型文件路径
    default_model_file = os.path.join(workspace_models_path, 'yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin')

    # ==========================================
    # 1. 声明启动参数
    # ==========================================
    conf_thres_arg = DeclareLaunchArgument('conf_thres', default_value='0.50', description='confidence threshold')
    show_image_arg = DeclareLaunchArgument('show_image', default_value='false', description='Whether to show detection image window')
    x_off_arg = DeclareLaunchArgument('x_offset', default_value='0.00', description='Translation in X')
    y_off_arg = DeclareLaunchArgument('y_offset', default_value='0.00', description='Translation in Y')
    z_off_arg = DeclareLaunchArgument('z_offset', default_value='0.00', description='Translation in Z')
    model_arg = DeclareLaunchArgument('model_filename', default_value=default_model_file, description='Model filename')
    class_file_arg = DeclareLaunchArgument('class_labels_file', default_value='', description='Path to .names or .txt file for class names')
    model_filename = LaunchConfiguration('model_filename')

    # ==========================================
    # 2. 判断模型类型
    # ==========================================

    # 1. 判断是否为 ONNX 模型 (后缀是否为 .onnx)
    # 拼接后的 Python 代码类似于: 'filename.onnx'.endswith('.onnx')
    is_onnx_model = PythonExpression(["'", model_filename, "'.lower().endswith('.onnx')"])

    # 2. 判断是否为 BPU Seg 模型 (不是 ONNX 且 包含 'seg')
    # 拼接后: not 'file.bin'.endswith('.onnx') and 'seg' in 'file.bin'
    is_bpu_seg_model = PythonExpression([
        "not '", model_filename, "'.lower().endswith('.onnx') and 'seg' in '", model_filename, "'"
    ])

    # 3. 判断是否为 BPU Detect 模型 (不是 ONNX 且 不含 'seg')
    # 拼接后: not 'file.bin'.endswith('.onnx') and 'seg' not in 'file.bin'
    is_bpu_detect_model = PythonExpression([
        "not '", model_filename, "'.lower().endswith('.onnx') and 'seg' not in '", model_filename, "'"
    ])

    # ==========================================
    # 3. RealSense 相机启动 (基础驱动)
    # ==========================================
    rs_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"),
                          "launch", 
                          "rs_launch.py")),
        launch_arguments={
            # === 1. 基础配置 ===
            'camera_name': 'realsense_d435i',
            'camera_namespace': 'camera',
            'initial_reset': 'false',
            
            # === 2. 核心功能 ===
            'align_depth.enable': 'true',     # 必须：深度对齐
            'enable_sync': 'true',            # 必须：时间同步
            'rgb_camera.color_profile': '1280,720,6', 
            'depth_module.depth_profile': '1280,720,6',

            # === 3. 日志 ===
            'log_level': 'warn'
        }.items()
    )

    # ==========================================
    # 4. 智能激活脚本 (改动点：循环检测 + 合并逻辑)
    # ==========================================
    # 这个脚本会一直运行，直到相机完全激活 (Active) 才会退出
    setup_script = """
        echo "[Launch] Step 1: Waiting for RealSense node..."
        # 1. 死循环等待节点出现
        until ros2 node list | grep "/camera/realsense_d435i" > /dev/null; do
            sleep 1
            echo "[Launch] Waiting for camera node..."
        done
        
        echo "[Launch] Step 2: Configuring camera..."
        # 2. 死循环尝试 Configure，直到成功
        until ros2 lifecycle set /camera/realsense_d435i configure; do
            sleep 2
            echo "[Launch] Retrying configure..."
        done

        echo "[Launch] Step 3: Activating camera..."
        # 3. 死循环尝试 Activate，直到成功
        until ros2 lifecycle set /camera/realsense_d435i activate; do
            sleep 2
            echo "[Launch] Retrying activate..."
        done
        
        echo "[Launch] Camera is ACTIVE! Triggering algorithms..."
        # 脚本成功结束，退出码 0，这将触发下面的事件处理器
        exit 0
    """

    camera_setup_cmd = ExecuteProcess(
        cmd=['bash', '-c', setup_script],
        output='screen'
    )

    # ==========================================
    # 5. 核心处理节点组 (改动点：打包但不立即运行)
    # ==========================================
    # 这里放所有的算法节点：detect, transform, viewer
    algorithm_nodes = GroupAction(
        actions=[
            LogInfo(msg="[Launch] Mode: ONNX Mode Activated!", condition=IfCondition(is_onnx_model)),
            LogInfo(msg="[Launch] Mode: BPU Segment Mode Activated!", condition=IfCondition(is_bpu_seg_model)),
            LogInfo(msg="[Launch] Mode: BPU Detect Mode Activated!", condition=IfCondition(is_bpu_detect_model)),
            
            # (A1) YOLO Detect Node(ONNX)
            Node(
                package='detect_yolov8_11_cpu',
                executable='yolo_detect_node',
                name='inference_node',
                output='screen',
                condition=IfCondition(is_onnx_model),
                parameters=[{
                    'conf_thres': LaunchConfiguration('conf_thres'),
                    'show_image': LaunchConfiguration('show_image'),
                    'model_filename': LaunchConfiguration('model_filename'),
                    'class_labels_file': LaunchConfiguration('class_labels_file') 
                }]
            ),

            # (A2) YOLO Detect Node(BIN)
            Node(
                package='detect_yolov5_bpu',
                executable='yolo_detect_node',
                name='inference_node',
                output='screen',
                condition=IfCondition(is_bpu_detect_model),
                parameters=[{
                    'conf_thres': LaunchConfiguration('conf_thres'),
                    'show_image': LaunchConfiguration('show_image'),
                    'model_filename': LaunchConfiguration('model_filename')
                }]
            ),

            # (A3) YOLO Segment Node
            Node(
                package='segment_yolov8_11_bpu',
                executable='yolo_seg_node',
                name='inference_node',
                output='screen',
                condition=IfCondition(is_bpu_seg_model),
                parameters=[{
                    'conf_thres': LaunchConfiguration('conf_thres'),
                    'show_image': LaunchConfiguration('show_image'),
                    'model_filename': LaunchConfiguration('model_filename'),
                }]
            ),

            # (B) Coord Transformer
            Node(
                package='coord_transformer',
                executable='transform_node',
                name='transform_node',
                output='screen',
                parameters=[{
                    'x_offset': LaunchConfiguration('x_offset'),
                    'y_offset': LaunchConfiguration('y_offset'),
                    'z_offset': LaunchConfiguration('z_offset')
                }]
            ),

            # (C) Image Viewer Talker
            Node(
                package='image_viewer',
                executable='image_viewer_talker',
                name='image_viewer_talker',
                output='screen'
            ),
        ]
    )

    # ==========================================
    # 6. 事件处理器 (改动点：核心逻辑)
    # ==========================================
    # 逻辑：当 camera_setup_cmd 进程退出（on_exit）时 -> 启动 algorithm_nodes
    event_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=camera_setup_cmd,
            on_exit=[algorithm_nodes]
        )
    )

    return LaunchDescription([
        # 0. 参数
        x_off_arg, y_off_arg, z_off_arg,
        conf_thres_arg, show_image_arg, model_arg,class_file_arg,
        
        # 1. 立即启动相机驱动 (此时是 inactive 状态)
        rs_camera_launch,
        
        # 2. 立即启动设置脚本 (开始循环检测)
        camera_setup_cmd,
        
        # 3. 注册事件：等脚本跑完了，再启动 YOLO 等节点
        event_handler
    ])