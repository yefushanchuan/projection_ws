import sys
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QGroupBox, 
                             QDoubleSpinBox, QCheckBox, QSplitter)
from PyQt5.QtCore import QProcess, Qt

class SystemDashboard(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.launch_process = None

    def initUI(self):
        self.setWindowTitle('ROS 2 投影系统控制台 (Projection Control)')
        self.resize(800, 600)
        
        main_layout = QVBoxLayout()

        # ========================================================
        # 区域 1: 系统启动控制 (Launch Control)
        # ========================================================
        launch_group = QGroupBox("1. 系统启动 (System Launch)")
        launch_group.setStyleSheet("QGroupBox { font-weight: bold; color: #2E86C1; }")
        launch_layout = QHBoxLayout()

        # 启动参数
        self.chk_show_image = QCheckBox("显示图像 (show_image)")
        self.chk_show_image.setChecked(True)
        launch_layout.addWidget(self.chk_show_image)

        # 启动/停止按钮
        self.btn_start = QPushButton("启动系统 (Start)")
        self.btn_start.setStyleSheet("background-color: #28B463; color: white; font-weight: bold; padding: 8px;")
        self.btn_start.clicked.connect(self.start_launch)
        
        self.btn_stop = QPushButton("停止系统 (Stop)")
        self.btn_stop.setStyleSheet("background-color: #C0392B; color: white; font-weight: bold; padding: 8px;")
        self.btn_stop.clicked.connect(self.stop_launch)
        self.btn_stop.setEnabled(False)

        launch_layout.addWidget(self.btn_start)
        launch_layout.addWidget(self.btn_stop)
        launch_group.setLayout(launch_layout)
        main_layout.addWidget(launch_group)

        # ========================================================
        # 区域 2: 坐标变换参数微调 (Param Tuning)
        # ========================================================
        param_group = QGroupBox("2. 坐标变换参数 (Real-time Tuning)")
        param_group.setStyleSheet("QGroupBox { font-weight: bold; color: #D35400; }")
        param_layout = QHBoxLayout()

        # 创建三个调节框
        self.spin_x = self.create_param_box("X Offset:", "x_offset", 0.0)
        self.spin_y = self.create_param_box("Y Offset:", "y_offset", 0.05) # 默认值设为你之前的0.05
        self.spin_z = self.create_param_box("Z Offset:", "z_offset", 0.0)

        param_layout.addLayout(self.spin_x)
        param_layout.addLayout(self.spin_y)
        param_layout.addLayout(self.spin_z)
        
        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # ========================================================
        # 区域 3: 运行日志 (Log)
        # ========================================================
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #17202A; color: #00FF00; font-family: Monospace; font-size: 10pt;")
        
        main_layout.addWidget(QLabel("系统日志 (System Log):"))
        main_layout.addWidget(self.log_text)

        self.setLayout(main_layout)

    def create_param_box(self, label_text, param_name, default_val):
        """辅助函数：创建参数调节组件"""
        layout = QVBoxLayout()
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        
        spin = QDoubleSpinBox()
        spin.setRange(-10.0, 10.0)   # 设置调节范围
        spin.setSingleStep(0.01)     # 步长 0.01
        spin.setDecimals(3)          # 保留3位小数
        spin.setValue(default_val)
        
        # 绑定信号：值改变时触发 param set
        # 使用 lambda 传递参数名
        spin.valueChanged.connect(lambda val: self.update_ros_param(param_name, val))
        
        layout.addWidget(label)
        layout.addWidget(spin)
        return layout

    def start_launch(self):
        """执行 ros2 launch"""
        show_img = "true" if self.chk_show_image.isChecked() else "false"
        
        # 获取当前 SpinBox 的值作为启动初始值
        x_off = str(self.spin_x.itemAt(1).widget().value())
        y_off = str(self.spin_y.itemAt(1).widget().value())
        z_off = str(self.spin_z.itemAt(1).widget().value())

        # 构造命令 (包含 source)
        cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source ~/realsense_ws/install/setup.bash && "
            "source ~/projection_ws/install/setup.bash && "
            f"ros2 launch cpp_launch system_launch.py "
            f"show_image:={show_img} "
            f"x_offset:={x_off} y_offset:={y_off} z_offset:={z_off}"
        )

        self.log_text.clear()
        self.log_text.append(">>> 正在启动系统...")
        self.log_text.append(f">>> 命令: {cmd}")

        self.launch_process = QProcess()
        self.launch_process.setProcessChannelMode(QProcess.MergedChannels)
        self.launch_process.readyReadStandardOutput.connect(self.read_output)
        self.launch_process.finished.connect(self.launch_finished)
        self.launch_process.start("bash", ["-c", cmd])

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_launch(self):
        """停止 launch"""
        if self.launch_process and self.launch_process.state() == QProcess.Running:
            self.log_text.append(">>> 正在发送停止信号...")
            # 这是一个比较强力的停止方式，确保杀死 ros2 launch 及其子进程
            # 简单的 terminate() 有时杀不掉子节点
            pid = self.launch_process.processId()
            subprocess.run(["kill", "-INT", str(pid)]) 

    def update_ros_param(self, param_name, value):
        """在后台调用 ros2 param set"""
        if not self.btn_stop.isEnabled():
            return # 系统没启动时不发送命令

        node_name = "/transform_node" # 你的 C++ 节点名
        
        # 构造命令：ros2 param set /transform_node x_offset 0.1
        # 注意：不要在主线程阻塞，开启一个新的简短进程去设置
        cmd = ["ros2", "param", "set", node_name, param_name, str(value)]
        
        self.log_text.append(f"[PARAM] Setting {param_name} -> {value}")
        
        # 使用 subprocess.Popen 异步执行，不卡界面
        subprocess.Popen(cmd)

    def read_output(self):
        data = self.launch_process.readAllStandardOutput()
        self.log_text.insertPlainText(bytes(data).decode('utf-8'))
        self.log_text.ensureCursorVisible()

    def launch_finished(self):
        self.log_text.append(">>> 系统已停止。")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SystemDashboard()
    ex.show()
    sys.exit(app.exec_())