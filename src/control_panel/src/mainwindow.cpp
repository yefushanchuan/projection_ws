#include "control_panel/mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , launch_process(nullptr)
    , ros_worker(nullptr)
{
    // 1. 初始化 Worker
    ros_worker = new RosWorker(this);
    ros_worker->start();

    // 2. 初始化 Process
    launch_process = new QProcess(this);
    connect(launch_process, &QProcess::readyReadStandardOutput, [this](){
        QByteArray data = launch_process->readAllStandardOutput();
        // 简单打印，保持原逻辑
        qDebug().noquote() << data; 
    });

    // 3. 构建界面 (保持原汁原味)
    setupUi();

    setWindowTitle("ROS 2 Control Panel");
    resize(300, 300); // 保持原来的尺寸
    
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusLabel = new QLabel("系统就绪");
    statusBar->addWidget(statusLabel);
}

MainWindow::~MainWindow() {
    // 1. 杀主进程
    if(launch_process && launch_process->state() == QProcess::Running) {
        launch_process->kill(); 
    }

    // 2. 停 Worker
    if(ros_worker) {
        ros_worker->quit();
        ros_worker->wait(50); // 稍微等一下即可
    }

    if(rclcpp::ok()) {
        rclcpp::shutdown();
    }

    // 3. 【核心优化】分离式后台清理
    // 这里的命令和 onStopClicked 基本一致，但去掉了 daemon start (退出程序了没必要启动它)
    QString cleanupCmd = 
        "pkill -9 -f detect_yolov5; "
        "pkill -9 -f transform_node; "
        "pkill -9 -f image_viewer; "
        "pkill -9 -f realsense; "
        "pkill -9 -f component_container; "
        "pkill -9 -f robot_state_publisher; "
        "rm -f /dev/shm/fastrtps_*; "
        "ros2 daemon stop";           // 退出时只需停止守护进程

    // 关键：startDetached 不会阻塞当前析构函数，窗口瞬间消失
    QProcess::startDetached("bash", QStringList() << "-c" << cleanupCmd);
}

void MainWindow::closeEvent(QCloseEvent *event) {
    this->hide();
    event->accept();
}

void MainWindow::onStartClicked()
{
    // 1. 界面立即反馈
    btn_start->setEnabled(false);
    btn_start->setText("启动中...");
    statusLabel->setText("正在启动 ROS 2 Launch...");
    
    // 【关键】刷新 UI，防止卡死
    qApp->processEvents();
    
    // 2. 获取参数
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";
    QString x_val = QString::number(spin_x->value(), 'f', 2);
    QString y_val = QString::number(spin_y->value(), 'f', 2);
    QString z_val = QString::number(spin_z->value(), 'f', 2);
    QString model_str = le_model_path->text().trimmed();

    // 3. 构造脚本
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

    // 4. 执行
    launch_process->start("bash", QStringList() << "-c" << script);

    // 5. 等待启动
    if (launch_process->waitForStarted(2000)) {
        btn_start->setText("系统运行中");
        btn_stop->setEnabled(true);
        chk_show_image->setEnabled(true); 
        
        le_model_path->setEnabled(false);
        btn_browse->setEnabled(false);
        
        statusLabel->setText("系统运行正常");
    } else {
        QMessageBox::critical(this, "启动失败", "无法启动 Bash 进程，请检查环境！");
        btn_start->setText("启动系统");
        btn_start->setEnabled(true);
        statusLabel->setText("启动失败");
    }
}

void MainWindow::onStopClicked()
{
    // 1. UI 立即反馈
    btn_stop->setEnabled(false);
    btn_stop->setText("停止中...");
    statusLabel->setText("正在清理系统资源...");
    
    // 强制刷新 UI，让按钮变灰，防止用户以为卡死
    qApp->processEvents();

    // 2. 终止 Launch 主进程
    if (launch_process->state() == QProcess::Running) {
        launch_process->terminate();
        // 这里不需要等太久，因为后面有强杀兜底
        launch_process->waitForFinished(100); 
    }
    
    // 3. 【核心优化】构建单条批处理指令
    // 逻辑：
    // (1) 先给 image_viewer 发 SIGINT (-2)，试图优雅关闭窗口
    // (2) Bash 内部睡眠 0.1 秒 (sleep 0.3)，给它时间析构
    // (3) 之后一并强杀所有相关节点 (-9)
    // (4) 清理共享内存 & 重启 Daemon
    
    QString cleanupCmd = 
        "pkill -2 -f image_viewer; "       // 1. 尝试优雅关闭
        "sleep 0.1; "                      // 2. 等待 300ms
        "pkill -9 -f detect_yolov5; "      // 3. 批量强杀其他
        "pkill -9 -f transform_node; "
        "pkill -9 -f realsense; "
        "pkill -9 -f component_container; "
        "pkill -9 -f robot_state_publisher; "
        "rm -f /dev/shm/fastrtps_*; "      // 4. 清理内存
        "ros2 daemon stop; "               // 5. 重置环境
        "ros2 daemon start";

    // 4. 执行命令 (阻塞式，但因为是一次性执行，效率很高)
    // 注意：主线程会卡这 0.3秒 + 执行时间，但因为前面 processEvents 了，用户无感
    QProcess::execute("bash", QStringList() << "-c" << cleanupCmd);

    // 5. 恢复界面状态
    btn_start->setEnabled(true);
    btn_start->setText("启动系统");
    
    btn_stop->setText("停 止 系 统");
    btn_stop->setEnabled(false);
    
    le_model_path->setEnabled(true);
    btn_browse->setEnabled(true);
    
    statusLabel->setText("系统已停止");
    qDebug() << "System stopped (Batch mode).";
}

void MainWindow::onParamChanged(const QString &name, double value) {
    if(ros_worker) {
        ros_worker->setParam(name.toStdString(), value);
    }
}

// 保持你原来的 SetupUI 完全不变
void MainWindow::setupUi() {
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // --- 系统设置区 ---
    QGroupBox *grpLaunch = new QGroupBox("系统设置", this);
    QVBoxLayout *layLaunch = new QVBoxLayout(grpLaunch);

    // 1. 模型选择
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

    // 2. 按钮区
    QHBoxLayout *layActions = new QHBoxLayout();

    chk_show_image = new QCheckBox("同时显示识别图像", this);
    chk_show_image->setChecked(true);
    connect(chk_show_image, &QCheckBox::clicked, [this](bool checked){
        if(launch_process->state() == QProcess::Running && ros_worker) {
            ros_worker->setParam("show_image", checked ? 1.0 : 0.0);
        }
    });
    layActions->addWidget(chk_show_image);

    layActions->addStretch();

    // 启动按钮 (保持原样式)
    btn_start = new QPushButton("启 动 系 统", this);
    btn_start->setMinimumWidth(100); 
    btn_start->setStyleSheet("background-color: green; color: white; font-weight: bold;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);
    layActions->addWidget(btn_start);

    // 停止按钮 (保持原样式)
    btn_stop = new QPushButton("停 止 系 统", this);
    btn_stop->setMinimumWidth(100); 
    btn_stop->setStyleSheet("background-color: red; color: white; font-weight: bold;");
    btn_stop->setEnabled(false);
    connect(btn_stop, &QPushButton::clicked, this, &MainWindow::onStopClicked);
    layActions->addWidget(btn_stop);

    layLaunch->addLayout(layActions);
    mainLayout->addWidget(grpLaunch);

    // --- 参数调节区 ---
    QGroupBox *grpParam = new QGroupBox("坐标偏移调节", this);
    QVBoxLayout *layParam = new QVBoxLayout(grpParam);

    auto createRow = [&](const QString &label, const QString &paramName, double defaultVal, QDoubleSpinBox* &spinBox) {
        QHBoxLayout *row = new QHBoxLayout();
        row->addWidget(new QLabel(label));
        row->addStretch(); 

        spinBox = new QDoubleSpinBox();
        spinBox->setRange(-1.0, 1.0); // 保持你原来的范围
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