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
    // QFont globalFont = this->font();
    // globalFont.setPointSize(10);
    // this->setFont(globalFont); 
    setupUi();
    
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusLabel = new QLabel("就绪");
    statusBar->addWidget(statusLabel);

    // 在 onStartClicked 中添加状态更新
    statusLabel->setText("系统启动中...");

    // 在 onStopClicked 中添加状态更新
    statusLabel->setText("系统已停止"); 

    // 设置一下窗口标题
    setWindowTitle("ROS 2 Control Panel");
    resize(300, 300);
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
    le_model_path->setPlaceholderText("Default : yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin");
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
    btn_start->setMinimumWidth(100); 
    btn_start->setStyleSheet("background-color: green; color: white; font-weight: bold;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);
    layActions->addWidget(btn_start);

    // (C) 停止按钮
    btn_stop = new QPushButton("停 止 系 统", this);
    btn_stop->setMinimumWidth(100); 
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
        
        row->addStretch(); 

        spinBox = new QDoubleSpinBox();
        spinBox->setRange(-1.0, 1.0);
        spinBox->setSingleStep(0.01);
        spinBox->setValue(defaultVal);
        spinBox->setSuffix(" 米"); 
        spinBox->setFixedWidth(100); 

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
    // 禁用按钮，显示进度
    btn_start->setEnabled(false);
    btn_start->setText("启动中...");
    
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";
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
    
    // 连接进程完成信号
    connect(launch_process, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus status){
        qDebug() << "Launch process finished with exit code:" << exitCode;
        if (exitCode != 0) {
            QMessageBox::warning(this, "警告", "启动失败，请检查系统状态！");
            // 恢复按钮状态
            btn_start->setEnabled(true);
            btn_start->setText("启 动 系 统");
            btn_stop->setEnabled(false);
        }
    });
    
    launch_process->start("bash", QStringList() << "-c" << script);
    
    // 等待进程启动
    if (launch_process->waitForStarted(3000)) {
        btn_start->setText("运行中");
        btn_stop->setEnabled(true);
        chk_show_image->setEnabled(true);
        le_model_path->setEnabled(false);
        btn_browse->setEnabled(false);
    } else {
        QMessageBox::warning(this, "错误", "无法启动系统进程！");
        btn_start->setEnabled(true);
        btn_start->setText("启 动 系 统");
    }
}

void MainWindow::onStopClicked()
{
    // 更新状态
    statusLabel->setText("正在停止系统...");
    
    // 1. 停止 Launch 进程
    if (launch_process && launch_process->state() == QProcess::Running) {
        launch_process->terminate();
        if (!launch_process->waitForFinished(1000)) {
            launch_process->kill();
            launch_process->waitForFinished(500);
        }
    }
    
    // 2. 清理后台进程
    QStringList processes = {
        "detect_yolov5", "transform_node", "image_viewer",
        "realsense2_camera_node", "component_container", "robot_state_publisher"
    };
    
    foreach (const QString &proc, processes) {
        QProcess killProcess;
        killProcess.start("pkill", QStringList() << "-9" << "-f" << proc);
        killProcess.waitForFinished(100);
    }
    
    // 3. 清理共享内存
    QProcess::execute("bash", QStringList() << "-c" << "rm -f /dev/shm/fastrtps_*");
    
    // 4. 重置 ROS 2 守护进程
    QProcess daemonStop;
    daemonStop.start("ros2", QStringList() << "daemon" << "stop");
    daemonStop.waitForFinished(500);
    
    // 5. 恢复界面状态
    btn_start->setEnabled(true);
    btn_start->setText("启 动 系 统");
    btn_stop->setEnabled(false);
    le_model_path->setEnabled(true);
    btn_browse->setEnabled(true);
    
    statusLabel->setText("系统已停止");
    qDebug() << "System stopped.";
}

void MainWindow::onParamChanged(const QString &name, double value) {
    if(ros_worker) {
        // 调用 worker 线程去发送 ROS 参数请求
        ros_worker->setParam(name.toStdString(), value);
    }
}