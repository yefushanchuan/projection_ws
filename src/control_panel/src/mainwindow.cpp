#include "control_panel/mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    // 1. 创建并启动 ROS 线程
    ros_worker = new RosWorker(this);
    ros_worker->start(); // 这行代码会让 run() 在新线程里跑起来

    // 2. 初始化 Launch 进程对象
    launch_process = new QProcess(this);
    connect(launch_process, &QProcess::readyReadStandardOutput, [this](){
        // 读取 Launch 输出并打印
        QByteArray data = launch_process->readAllStandardOutput();
        qDebug().noquote() << data; 
    });

    // 3. 构建界面
    setupUi();
    
    // 设置一下窗口标题
    setWindowTitle("ROS 2 Control Panel");
    resize(400, 400);
}

MainWindow::~MainWindow() {
    // --- 退出时的清理顺序很重要 ---

    // 1. 先杀掉 Launch 进程
    if(launch_process->state() == QProcess::Running) {
        launch_process->terminate();
        launch_process->waitForFinished(1000); // 等待最多1秒
    }

    // 2. 关闭 ROS
    // rclcpp::shutdown 会让 worker 里的 spin() 退出循环
    if(rclcpp::ok()) {
        rclcpp::shutdown();
    }

    // 3. 等待线程结束
    ros_worker->quit();
    ros_worker->wait();
}

void MainWindow::setupUi() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // --- Launch 控制区 ---
    QGroupBox *grpLaunch = new QGroupBox("系统控制", this);
    QVBoxLayout *layLaunch = new QVBoxLayout(grpLaunch);
    
    chk_show_image = new QCheckBox("开启检测窗口 (show_image)", this);
    // 默认选中
    chk_show_image->setChecked(true); 
    layLaunch->addWidget(chk_show_image);

    // !!! 新增：连接 CheckBox 的点击信号，支持运行时切换 !!!
    connect(chk_show_image, &QCheckBox::clicked, [this](bool checked){
        // 只有当 Launch 已经启动后，点击才发送参数请求
        if(launch_process->state() == QProcess::Running && ros_worker) {
            // 发送 1.0 代表 true, 0.0 代表 false
            ros_worker->setParam("show_image", checked ? 1.0 : 0.0);
        }
    });

    QHBoxLayout *layButtons = new QHBoxLayout();
    btn_start = new QPushButton("启动系统", this);
    btn_start->setStyleSheet("background-color: green; color: white;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);

    btn_stop = new QPushButton("停止系统", this);
    btn_stop->setStyleSheet("background-color: red; color: white;");
    btn_stop->setEnabled(false);
    connect(btn_stop, &QPushButton::clicked, this, &MainWindow::onStopClicked);

    layButtons->addWidget(btn_start);
    layButtons->addWidget(btn_stop);
    layLaunch->addLayout(layButtons);
    mainLayout->addWidget(grpLaunch);

    // --- 参数调节区 (Offset) ---
    QGroupBox *grpParam = new QGroupBox("坐标偏移调节", this);
    QVBoxLayout *layParam = new QVBoxLayout(grpParam);

    auto createRow = [&](const QString &label, const QString &paramName, double defaultVal, QDoubleSpinBox* &spinBox) {
        QHBoxLayout *row = new QHBoxLayout();
        row->addWidget(new QLabel(label));
        spinBox = new QDoubleSpinBox();
        spinBox->setRange(-5.0, 5.0); // 根据实际需要调整范围
        spinBox->setSingleStep(0.01);  // 步长
        spinBox->setValue(defaultVal);
        row->addWidget(spinBox);
        layParam->addLayout(row);

        // 值改变 -> 触发 setParam
        connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            [this, paramName](double val){
                this->onParamChanged(paramName, val);
            });
    };

    createRow("X Offset (左右):", "x_offset", 0.0, spin_x);
    createRow("Y Offset (上下):", "y_offset", 0.0, spin_y); // 注意：你的 launch 默认好像是 0.05
    createRow("Z Offset (前后):", "z_offset", 0.0, spin_z);

    mainLayout->addWidget(grpParam);
}

void MainWindow::onStartClicked()
{
    // 获取当前 CheckBox 的状态作为启动参数
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";
    QString x_val = QString::number(spin_x->value(), 'f', 2);
    QString y_val = QString::number(spin_y->value(), 'f', 2);
    QString z_val = QString::number(spin_z->value(), 'f', 2);

    // 构造命令：同时传入 show_image 和 初始的 offset
    QString script = QString("source /opt/ros/humble/setup.bash && "
                             "ros2 launch cpp_launch system_launch.py " 
                             "show_image:=%1 "
                             "x_offset:=%2 "
                             "y_offset:=%3 "
                             "z_offset:=%4")
                             .arg(show_img_val, x_val, y_val, z_val);

    qDebug() << "Executing:" << script;

    launch_process->start("bash", QStringList() << "-c" << script);

    btn_start->setEnabled(false);
    btn_stop->setEnabled(true);
    
    // !!! 关键修改：启动后不要禁用 CheckBox，允许用户运行时开关窗口 !!!
    chk_show_image->setEnabled(true); 
}

void MainWindow::onStopClicked()
{
    launch_process->terminate();
    
    // 使用 QStringList 分离命令和参数
    // "pkill" 是命令， "-f" 和 "进程名" 是参数
    QProcess::execute("pkill", QStringList() << "-f" << "detect_yolov5");
    QProcess::execute("pkill", QStringList() << "-f" << "transform_node");
    QProcess::execute("pkill", QStringList() << "-f" << "image_viewer");
    QProcess::execute("pkill", QStringList() << "-f" << "camera");
    QProcess::execute("pkill", QStringList() << "-f" << "component_container");
    QProcess::execute("pkill", QStringList() << "-f" << "robot_state_publisher");

    btn_start->setEnabled(true);
    btn_stop->setEnabled(false);
}

void MainWindow::onParamChanged(const QString &name, double value) {
    if(ros_worker) {
        ros_worker->setParam(name.toStdString(), value);
    }
}