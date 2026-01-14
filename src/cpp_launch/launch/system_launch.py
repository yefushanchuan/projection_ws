import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, DeclareLaunchArgument, GroupAction, LogInfo
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory

def generate_launch_description():
    # ==========================================
    # 0. 声明启动参数 (保持原样)
    # ==========================================
    conf_thres_arg = DeclareLaunchArgument('conf_thres', default_value='0.35', description='YOLOv5 confidence threshold')
    show_image_arg = DeclareLaunchArgument('show_image', default_value='false', description='Whether to show detection image window')
    x_off_arg = DeclareLaunchArgument('x_offset', default_value='0.00', description='Translation in X')
    y_off_arg = DeclareLaunchArgument('y_offset', default_value='0.00', description='Translation in Y')
    z_off_arg = DeclareLaunchArgument('z_offset', default_value='0.00', description='Translation in Z')
    model_arg = DeclareLaunchArgument('model_filename', default_value='yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin', description='Model filename')

    # ==========================================
    # 1. RealSense 相机启动 (基础驱动)
    # ==========================================
    rs_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"),
                          "launch", 
                          "rs_launch.py")),
        launch_arguments={
            # 1. 节点名称和命名空间
            'camera_name': 'realsense_d435i',          
            'camera_namespace': 'camera',     
            
            # 2. 复位设置
            'initial_reset': 'false',
            
            # 3. 对齐和同步 (正确)
            'align_depth.enable': 'true',     
            'enable_sync': 'true',            
            
            # 4. 分辨率配置 (修正参数名，注意使用逗号分隔)
            'rgb_camera.color_profile': '1280,720,30',   
            'depth_module.depth_profile': '1280,720,30', 
            
            # 5. 关闭不需要的功能 (显式关闭以节省资源)
            'pointcloud.enable': 'false',
            'enable_gyro': 'false',
            'enable_accel': 'false',
            
            # 6. 日志级别
            'log_level': 'warn'
        }.items()
    )

    # ==========================================
    # 2. 智能激活脚本 (改动点：循环检测 + 合并逻辑)
    # ==========================================
    # 这个脚本会一直运行，直到相机完全激活 (Active) 才会退出
    setup_script = """
        echo "[Launch] Step 1: Waiting for RealSense node..."
        # 1. 死循环等待节点出现
        until ros2 node list | grep -q "/camera/realsense_d435i"; do
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
    # 3. 核心处理节点组 (改动点：打包但不立即运行)
    # ==========================================
    # 这里放所有的算法节点：detect, transform, viewer
    algorithm_nodes = GroupAction(
        actions=[
            LogInfo(msg="[Launch] Camera ready! Starting Deep Learning nodes..."),
            
            # (A) YOLO Detect Node
            Node(
                package='detect',
                executable='detect_yolov5',
                name='detect_yolov5_node',
                output='screen',
                parameters=[{
                    'conf_thres': LaunchConfiguration('conf_thres'),
                    'show_image': LaunchConfiguration('show_image'),
                    'model_filename': LaunchConfiguration('model_filename')
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

            # (D) Image Viewer Listener
            Node(
                package='image_viewer',
                executable='image_viewer_listener',
                name='image_viewer_listener',
                output='screen'
            ),
        ]
    )

    # ==========================================
    # 4. 事件处理器 (改动点：核心逻辑)
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
        conf_thres_arg, show_image_arg, model_arg,
        
        # 1. 立即启动相机驱动 (此时是 inactive 状态)
        rs_camera_launch,
        
        # 2. 立即启动设置脚本 (开始循环检测)
        camera_setup_cmd,
        
        # 3. 注册事件：等脚本跑完了，再启动 YOLO 等节点
        event_handler
    ])