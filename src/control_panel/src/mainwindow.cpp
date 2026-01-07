#include "control_panel/mainwindow.h"
#include <QDebug>
#include <QHBoxLayout>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // 1. 初始化 ROS 线程
    ros_worker = new RosWorker();
    ros_worker->start(); // 启动 ROS spin 线程

    // 2. 初始化 Launch 进程对象
    launch_process = new QProcess(this);
    connect(launch_process, &QProcess::readyReadStandardOutput, [this](){
        QByteArray data = launch_process->readAllStandardOutput();
        qDebug() << "[Launch Log]" << data; // 这里简单打印到控制台，你也可以显示到 TextEdit
    });

    // 3. 构建界面
    setupUi();
}

MainWindow::~MainWindow() {
    // 退出时清理
    ros_worker->terminate();
    rclcpp::shutdown();
    if(launch_process->state() == QProcess::Running) {
        launch_process->terminate();
    }
}

void MainWindow::setupUi() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // --- Launch 控制区 ---
    QGroupBox *grpLaunch = new QGroupBox("系统控制", this);
    QHBoxLayout *layLaunch = new QHBoxLayout(grpLaunch);
    
    // 1. 创建复选框
    chk_show_image = new QCheckBox("显示检测画面 (YOLO Monitor)", this);
    chk_show_image->setChecked(false); // 默认不勾选，对应 launch 中的 default_value='false'
    
    // 2. 把它加到布局里
    layLaunch->addWidget(chk_show_image);

    // 3. 按钮行 (为了美观，把按钮放在一行)
    QHBoxLayout *layButtons = new QHBoxLayout();
    btn_start = new QPushButton("启动系统", this);
    btn_start->setStyleSheet("background-color: green; color: white; height: 40px;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);

    btn_stop = new QPushButton("停止系统", this);
    btn_stop->setStyleSheet("background-color: red; color: white; height: 40px;");
    btn_stop->setEnabled(false);
    connect(btn_stop, &QPushButton::clicked, this, &MainWindow::onStopClicked);

    layButtons->addWidget(btn_start);
    layButtons->addWidget(btn_stop);

    // 把按钮行加到主控制区
    layLaunch->addLayout(layButtons);
    mainLayout->addWidget(grpLaunch);

    // --- 参数调节区 ---
    QGroupBox *grpParam = new QGroupBox("实时坐标变换 (Native C++ API)", this);
    QVBoxLayout *layParam = new QVBoxLayout(grpParam);

    // 辅助 lambda 用于创建一行控件
    auto createRow = [&](const QString &label, const QString &paramName, double defaultVal, QDoubleSpinBox* &spinBox) {
        QHBoxLayout *row = new QHBoxLayout();
        row->addWidget(new QLabel(label));
        spinBox = new QDoubleSpinBox();
        spinBox->setRange(-10.0, 10.0);
        spinBox->setSingleStep(0.01);
        spinBox->setValue(defaultVal);
        row->addWidget(spinBox);
        layParam->addLayout(row);

        // 连接信号: 值改变 -> 调用 ROS 设置
        connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            [this, paramName](double val){
                this->onParamChanged(paramName, val);
            });
    };

    createRow("X Offset:", "x_offset", 0.0, spin_x);
    createRow("Y Offset:", "y_offset", 0.05, spin_y);
    createRow("Z Offset:", "z_offset", 0.0, spin_z);

    mainLayout->addWidget(grpParam);
}

void MainWindow::onStartClicked()
{
    // 1. 获取复选框状态，转换为字符串 "true" 或 "false"
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";

    // 2. 拼接 Launch 参数
    // 格式: show_image:=true
    QString param_arg = QString("show_image:=%1").arg(show_img_val);

    // 3. 构造完整的脚本
    // 注意：在 system_launch.py 后面加上空格和参数
    QString script = "source /opt/ros/humble/setup.bash && "
                     "source ~/realsense_ws/install/setup.bash && "
                     "source ~/projection_ws/install/setup.bash && "
                     "ros2 launch cpp_launch system_launch.py " + param_arg;

    // 调试打印，方便查看拼出来的命令对不对
    qDebug() << "Executing Launch Command:" << script;

    // 4. 启动进程
    QString program = "bash";
    QStringList arguments;
    arguments << "-c" << script;

    launch_process->start(program, arguments);

    // 5. 更新按钮状态
    btn_start->setEnabled(false);
    btn_stop->setEnabled(true);
    // 启动后建议禁用复选框，防止用户在运行时修改（因为 Launch 参数只在启动时生效）
    chk_show_image->setEnabled(false); 
}

void MainWindow::onStopClicked()
{
    launch_process->terminate();
    
    btn_start->setEnabled(true);
    btn_stop->setEnabled(false);
    
    // 恢复复选框可用
    chk_show_image->setEnabled(true);
}

void MainWindow::onParamChanged(const QString &name, double value) {
    // 核心优化：直接调用内存中的 ROS Client，无需创建进程
    // 延迟 < 1ms
    ros_worker->setParam(name.toStdString(), value);
}