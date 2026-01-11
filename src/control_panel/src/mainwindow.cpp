#include "control_panel/mainwindow.h"
#include <QFileDialog> 

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
    resize(400, 300);
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

    // --- 系统设置区 ---
    QGroupBox *grpLaunch = new QGroupBox("系统设置", this);
    QVBoxLayout *layLaunch = new QVBoxLayout(grpLaunch);

    // ==========================================
    // 1. 第一排：模型选择
    // ==========================================
    QHBoxLayout *layModel = new QHBoxLayout();
    layModel->addWidget(new QLabel("模型文件:"));

    le_model_path = new QLineEdit();
    le_model_path->setPlaceholderText("yolov5n_tag_v7.0_detect_640x640_bayese_nv12.bin");
    layModel->addWidget(le_model_path);

    btn_browse = new QPushButton("浏览...");
    btn_browse->setMaximumWidth(50); 
    connect(btn_browse, &QPushButton::clicked, [this](){
        QString fileName = QFileDialog::getOpenFileName(
            this, "选择模型文件", "/home/sunrise", "Model Files (*.bin);;All Files (*)"
        );
        if (!fileName.isEmpty()) {
            le_model_path->setText(fileName);
        }
    });
    layModel->addWidget(btn_browse);
    layLaunch->addLayout(layModel);

    // ==========================================
    // 2. 第二排：检测开关 + 启动 + 停止
    // ==========================================
    QHBoxLayout *layActions = new QHBoxLayout();

    // (A) 检测窗口开关
    chk_show_image = new QCheckBox("同时显示识别图像", this);
    chk_show_image->setChecked(true);
    // 动态切换连接
    connect(chk_show_image, &QCheckBox::clicked, [this](bool checked){
        if(launch_process->state() == QProcess::Running && ros_worker) {
            ros_worker->setParam("show_image", checked ? 1.0 : 0.0);
        }
    });
    layActions->addWidget(chk_show_image);

    layActions->addStretch();

    // (B) 启动按钮
    btn_start = new QPushButton("启 动 系 统", this);
    btn_start->setMinimumWidth(150); 
    btn_start->setStyleSheet("background-color: green; color: white; font-weight: bold;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);
    layActions->addWidget(btn_start);

    // (C) 停止按钮
    btn_stop = new QPushButton("停 止 系 统", this);
    btn_stop->setMinimumWidth(150); 
    btn_stop->setStyleSheet("background-color: red; color: white; font-weight: bold;");
    btn_stop->setEnabled(false);
    connect(btn_stop, &QPushButton::clicked, this, &MainWindow::onStopClicked);
    layActions->addWidget(btn_stop);

    // 将这一排加入到 GroupBox
    layLaunch->addLayout(layActions);

    mainLayout->addWidget(grpLaunch);

    // ==========================================
    // 3. 参数调节区
    // ==========================================
    QGroupBox *grpParam = new QGroupBox("坐标偏移调节", this);
    QVBoxLayout *layParam = new QVBoxLayout(grpParam);

    auto createRow = [&](const QString &label, const QString &paramName, double defaultVal, QDoubleSpinBox* &spinBox) {
        QHBoxLayout *row = new QHBoxLayout();
        row->addWidget(new QLabel(label));
        spinBox = new QDoubleSpinBox();
        spinBox->setRange(-1.0, 1.0);
        spinBox->setSingleStep(0.01);
        spinBox->setValue(defaultVal);
        row->addWidget(spinBox);
        layParam->addLayout(row);

        connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            [this, paramName](double val){
                this->onParamChanged(paramName, val);
            });
    };

    createRow("X Offset (正即相机相对投影向左):", "x_offset", 0.00, spin_x);
    createRow("Y Offset (正即相机相对投影向上):", "y_offset", 0.00, spin_y);
    createRow("Z Offset (正即相机相对投影向后):", "z_offset", 0.00, spin_z);

    mainLayout->addWidget(grpParam);
}

void MainWindow::onStartClicked()
{
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";
    
    // !!! 核心修改：加 'f', 2 !!!
    // 强制把 0 转成 "0.00"，让 ROS 识别为 double 类型，防止 crash
    QString x_val = QString::number(spin_x->value(), 'f', 2);
    QString y_val = QString::number(spin_y->value(), 'f', 2);
    QString z_val = QString::number(spin_z->value(), 'f', 2);

    QString model_str = le_model_path->text().trimmed();

    QString script = QString("source /opt/ros/humble/setup.bash && "
                             "ros2 launch cpp_launch system_launch.py " 
                             "show_image:=%1 "
                             "x_offset:=%2 "
                             "y_offset:=%3 "
                             "z_offset:=%4 ")
                             .arg(show_img_val, x_val, y_val, z_val);

    if (!model_str.isEmpty()) {
        script += QString("model_filename:='%1'").arg(model_str);
    }
    
    qDebug() << "Executing:" << script;

    launch_process->start("bash", QStringList() << "-c" << script);

    btn_start->setEnabled(false);
    btn_stop->setEnabled(true);
    
    chk_show_image->setEnabled(true); 
    
    // 建议启动后锁定模型选择，防止误触
    le_model_path->setEnabled(false);
    btn_browse->setEnabled(false);
}

void MainWindow::onStopClicked()
{
    // 1. 停止 Launch 进程
    if (launch_process->state() == QProcess::Running) {
        launch_process->terminate();
        launch_process->waitForFinished(500); // 稍微等一下
    }
    
    // 2. 暴力清理所有相关后台进程 (建议加上 -9 强制杀死，防止进程卡在后台)
    // pkill -9 表示 SIGKILL，立即处决，不给进程犹豫的机会
    QProcess::execute("pkill", QStringList() << "-9" << "-f" << "detect_yolov5");
    QProcess::execute("pkill", QStringList() << "-9" << "-f" << "transform_node");
    QProcess::execute("pkill", QStringList() << "-9" << "-f" << "image_viewer");
    QProcess::execute("pkill", QStringList() << "-9" << "-f" << "realsense"); 
    QProcess::execute("pkill", QStringList() << "-9" << "-f" << "component_container");
    QProcess::execute("pkill", QStringList() << "-9" << "-f" << "robot_state_publisher");

    // ============================================================
    // !!! 核心修复点：清理共享内存 (Shared Memory) !!!
    // ============================================================
    // 这一步专门解决 [RTPS_TRANSPORT_SHM Error]
    // 因为 rm 支持通配符(*)，必须通过 bash -c 来执行
    QProcess::execute("bash", QStringList() << "-c" << "rm -f /dev/shm/fastrtps_*");

    // 3. 重置 ROS 2 守护进程 (清除节点重名缓存)
    QProcess::execute("ros2", QStringList() << "daemon" << "stop");
    QProcess::execute("ros2", QStringList() << "daemon" << "start"); // 顺手重启一下更好

    // 4. 恢复按钮状态
    btn_start->setEnabled(true);
    btn_stop->setEnabled(false);
    
    // 解锁模型选择
    le_model_path->setEnabled(true);
    btn_browse->setEnabled(true);
    
    qDebug() << "System stopped.";
}

void MainWindow::onParamChanged(const QString &name, double value) {
    if(ros_worker) {
        // 调用 worker 线程去发送 ROS 参数请求
        ros_worker->setParam(name.toStdString(), value);
    }
}