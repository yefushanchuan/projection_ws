import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory

def generate_launch_description():
    # ==========================================
    # 0. 声明启动参数
    # ==========================================
    conf_thres_arg = DeclareLaunchArgument('conf_thres', default_value='0.35', description='YOLOv5 confidence threshold')
    show_image_arg = DeclareLaunchArgument('show_image', default_value='false', description='Whether to show detection image window')
    x_off_arg = DeclareLaunchArgument('x_offset', default_value='0.00', description='Translation in X')
    y_off_arg = DeclareLaunchArgument('y_offset', default_value='0.00', description='Translation in Y')
    z_off_arg = DeclareLaunchArgument('z_offset', default_value='0.00', description='Translation in Z')
    model_arg = DeclareLaunchArgument('model_filename', default_value='yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin', description='Model filename inside detect/models folder')

    # ==========================================
    # 1. RealSense 相机启动
    # ==========================================
    rs_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"),
                          "launch", 
                          "rs_launch.py")),
        launch_arguments={
            'name': 'camera',
            'namespace': 'camera',
            # 'initial_reset': 'true',  # 保持这个开启，防止硬件卡死
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'enable_color': 'true',
            'enable_depth': 'true',
            'log_level': 'warn' 
        }.items()
    )

    # ==========================================
    # 2. 智能 Configure (带重试机制)
    # ==========================================
    # 修复点：使用三引号包裹脚本，确保语法正确
    configure_script = """
        echo "Waiting for RealSense node to appear..."
        for i in {1..30}; do
            if ros2 node list | grep -q "/camera/camera"; then
                echo "Node found! Configuring..."
                sleep 2
                ros2 lifecycle set /camera/camera configure
                exit 0
            fi
            sleep 1
        done
        echo "Timeout waiting for camera node!"
        exit 1
    """

    configure_camera_cmd = ExecuteProcess(
        cmd=['bash', '-c', configure_script],
        output='screen'
    )

    # ==========================================
    # 3. 智能 Activate (带等待机制)
    # ==========================================
    activate_script = """
        echo "Waiting for camera to be Configured (inactive state)..."
        
        # 1. 等待 Configure 完成
        while ! ros2 lifecycle get /camera/camera | grep -q "inactive"; do
            sleep 0.5
        done

        echo "Node is Configured."
        
        # 2. 修改点：既然关闭了 initial_reset，就不需要等 10 秒了
        # 给 1-2 秒缓冲即可，确保状态切换稳定
        sleep 2
        
        # 3. 开始激活
        echo "Attempting to Activate..."
        for i in {1..10}; do
            if ros2 lifecycle set /camera/camera activate; then
                echo "Activation successful!"
                exit 0
            fi
            echo "Activate failed, retrying in 1s..."
            sleep 1
        done
        echo "Activation Timed Out!"
        exit 1
    """

    activate_camera_cmd = ExecuteProcess(
        cmd=['bash', '-c', activate_script],
        output='screen'
    )

    # ==========================================
    # 4. Detect YOLOv5
    # ==========================================
    detect_node = Node(
        package='detect',
        executable='detect_yolov5',
        name='detect_yolov5_node',
        output='screen',
        parameters=[{
            'conf_thres': LaunchConfiguration('conf_thres'),
            'show_image': LaunchConfiguration('show_image'),
            'model_filename': LaunchConfiguration('model_filename')
        }]
    )

    # ==========================================
    # 6. Coord Transformer
    # ==========================================
    transform_node = Node(
        package='coord_transformer',
        executable='transform_node',
        name='transform_node',
        output='screen',
        parameters=[{
            'x_offset': LaunchConfiguration('x_offset'),
            'y_offset': LaunchConfiguration('y_offset'),
            'z_offset': LaunchConfiguration('z_offset')
        }]
    )

    # ==========================================
    # 7. Image Viewer Talker
    # ==========================================
    viewer_talker_node = Node(
        package='image_viewer',
        executable='image_viewer_talker',
        name='image_viewer_talker',
        output='screen'
    )

    # ==========================================
    # 8. Image Viewer Listener
    # ==========================================
    viewer_listener_node = Node(
        package='image_viewer',
        executable='image_viewer_listener',
        name='image_viewer_listener',
        output='screen'
    )

    return LaunchDescription([
        x_off_arg,
        y_off_arg,
        z_off_arg,
        conf_thres_arg,
        show_image_arg,
        model_arg,
        rs_camera_launch,
        configure_camera_cmd,
        activate_camera_cmd,
        detect_node,
        transform_node,
        viewer_talker_node,
        viewer_listener_node
    ])