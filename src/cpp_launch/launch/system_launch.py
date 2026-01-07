import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python import get_package_share_directory

def generate_launch_description():
    # ==========================================
    # 0. 声明启动参数 (Configuration)
    # ==========================================
    # 定义 conf_thres 参数，默认值 0.35
    conf_thres_arg = DeclareLaunchArgument('conf_thres', default_value='0.35', description='YOLOv5 confidence threshold')

    # 定义 show_image 参数，默认值 false (注意：Launch文件中布尔值通常作为字符串传递)
    show_image_arg = DeclareLaunchArgument('show_image', default_value='false', description='Whether to show detection image window')

    x_off_arg = DeclareLaunchArgument('x_offset', default_value='0.0', description='Translation in X')
    y_off_arg = DeclareLaunchArgument('y_offset', default_value='0.05', description='Translation in Y')
    z_off_arg = DeclareLaunchArgument('z_offset', default_value='0.0', description='Translation in Z')
    # ==========================================
    # 1. RealSense 相机启动
    # ==========================================
    # 对应命令: ros2 launch realsense2_camera rs_launch.py enable_rgbd:=true ...
    rs_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"),
                          "launch", 
                          "rs_launch.py")),
        launch_arguments={
            'enable_rgbd': 'true',
            'enable_sync': 'true',
            'align_depth.enable': 'true',
            'enable_color': 'true',
            'enable_depth': 'true'
        }.items()
    )

    # ==========================================
    # 2 & 3. Lifecycle 生命周期管理
    # ==========================================
    # 对应命令: ros2 lifecycle set /camera/camera configure
    # 设置延时 5 秒，等待相机节点加载完毕后再执行 configure
    configure_camera_cmd = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'lifecycle', 'set', '/camera/camera', 'configure'],
                output='screen'
            )
        ]
    )

    # 对应命令: ros2 lifecycle set /camera/camera activate
    # 设置延时 10 秒 (Configure 之后 5 秒)，执行 activate
    activate_camera_cmd = TimerAction(
        period=10.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'lifecycle', 'set', '/camera/camera', 'activate'],
                output='screen'
            )
        ]
    )

    # ==========================================
    # 4. Detect YOLOv5 (projection_ws)
    # ==========================================
    # 对应命令: ros2 run detect detect_yolov5
    detect_node = Node(
        package='detect',
        executable='detect_yolov5',
        name='detect_yolov5_node',
        output='screen',
        parameters=[{
            'conf_thres': LaunchConfiguration('conf_thres'),
            'show_image': LaunchConfiguration('show_image')
        }]
    )

    # ==========================================
    # 6. Coord Transformer (projection_ws)
    # ==========================================
    # 对应命令: ros2 run coord_transformer transform_node
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
    # 7. Image Viewer Talker (projection_ws)
    # ==========================================
    # 对应命令: ros2 run image_viewer image_viewer_talker
    viewer_talker_node = Node(
        package='image_viewer',
        executable='image_viewer_talker',
        name='image_viewer_talker',
        output='screen'
    )

    # ==========================================
    # 8. Image Viewer Listener (projection_ws)
    # ==========================================
    # 对应命令: ros2 run image_viewer image_viewer_listener
    viewer_listener_node = Node(
        package='image_viewer',
        executable='image_viewer_listener',
        name='image_viewer_listener',
        output='screen'
    )

    # ==========================================
    # 返回 Launch 描述
    # ==========================================
    return LaunchDescription([
        x_off_arg,
        y_off_arg,
        z_off_arg,
        conf_thres_arg,
        show_image_arg, 
        rs_camera_launch,
        configure_camera_cmd,
        activate_camera_cmd,
        detect_node,
        transform_node,
        viewer_talker_node,
        viewer_listener_node
    ])