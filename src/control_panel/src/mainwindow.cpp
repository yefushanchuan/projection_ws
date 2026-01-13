#include "control_panel/mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , launch_process(nullptr) // 初始化指针，好习惯
    , ros_worker(nullptr)
{
    // 1. 【核心】先构建界面，防止 worker 线程访问未初始化的 UI 导致崩溃
    setupUi();

    // 2. 初始化状态栏
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusLabel = new QLabel("系统就绪");
    statusBar->addWidget(statusLabel);

    // 3. 启动 ROS 线程
    ros_worker = new RosWorker(this);
    ros_worker->start(); 

    // 4. 初始化进程对象并连接信号
    launch_process = new QProcess(this);
    
    // 读取标准输出 (stdout)
    connect(launch_process, &QProcess::readyReadStandardOutput, [this](){
        QByteArray data = launch_process->readAllStandardOutput();
        QString output = QString::fromLocal8Bit(data);
        
        // 简单过滤，只打印关键信息，避免刷屏
        if(output.contains("ERROR") || output.contains("died")) {
             qWarning().noquote() << "[Launch ERROR]" << output;
        } else {
             qDebug().noquote() << "[Launch]" << output; 
        }
    });

    // 读取错误输出 (stderr)
    connect(launch_process, &QProcess::readyReadStandardError, [this](){
         QByteArray data = launch_process->readAllStandardError();
         qWarning().noquote() << "[Launch STDERR]" << data;
    });

    // 处理进程启动失败的情况
    connect(launch_process, &QProcess::errorOccurred, [this](QProcess::ProcessError error){
        if (error == QProcess::FailedToStart) {
            QMessageBox::critical(this, "系统错误", "无法启动 Bash 环境，请检查系统路径！");
            statusLabel->setText("环境错误");
        }
    });

    // 窗口基本设置
    setWindowTitle("ROS 2 Control Panel");
    resize(300, 300); 
}

MainWindow::~MainWindow() {
    if(launch_process && launch_process->state() == QProcess::Running) {
        launch_process->kill(); 
    }

    if(ros_worker) {
        ros_worker->quit();
        if(!ros_worker->wait(100)) ros_worker->terminate();
    }

    if(rclcpp::ok()) {
        rclcpp::shutdown();
    }

    // 退出时的清理命令 (不需要重启 daemon，追求速度)
    QString cleanupCmd = 
        "pkill -9 -f detect_yolov5; "
        "pkill -9 -f transform_node; "
        "pkill -9 -f image_viewer; "
        "pkill -9 -f realsense; "
        "pkill -9 -f component_container; "
        "pkill -9 -f system_launch.py; "
        "rm -f /dev/shm/fastrtps_*; "
        "ros2 daemon stop"; 

    QProcess::startDetached("bash", QStringList() << "-c" << cleanupCmd);
}

void MainWindow::closeEvent(QCloseEvent *event) {
    // 隐藏窗口给用户“立即关闭”的感觉
    this->hide();
    event->accept();
}

void MainWindow::onStartClicked()
{
    // 1. 界面立即反馈
    btn_start->setEnabled(false);
    btn_start->setText("启动中...");
    statusLabel->setText("正在启动 ROS 2 Launch...");
    
    // 【关键】强制刷新 UI，防止界面卡死
    qApp->processEvents();

    // 2. 获取参数
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";
    
    // 强制保留两位小数，防止格式错误
    QString x_val = QString::number(spin_x->value(), 'f', 2);
    QString y_val = QString::number(spin_y->value(), 'f', 2);
    // 【已修复】这里原先是 spin_y，现已修正为 spin_z
    QString z_val = QString::number(spin_z->value(), 'f', 2); 

    QString model_str = le_model_path->text().trimmed();

    // 3. 构造启动脚本
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
    
    qDebug() << "Executing Script:" << script;

    // 4. 启动进程
    launch_process->start("bash", QStringList() << "-c" << script);

    // 5. 等待启动结果 (阻塞最多 2000ms)
    if (launch_process->waitForStarted(2000)) {
        // --- 成功 ---
        btn_start->setText("系统运行中");
        
        btn_stop->setEnabled(true);
        chk_show_image->setEnabled(true); 
        
        // 运行时锁定模型选择
        le_model_path->setEnabled(false);
        btn_browse->setEnabled(false);
        
        statusLabel->setText("系统运行正常");
        
    } else {
        // --- 失败 ---
        // 错误信息已在 errorOccurred 信号中处理，这里主要复位 UI
        btn_start->setText("启动系统");
        btn_start->setEnabled(true);
        statusLabel->setText("启动失败");
    }
}

void MainWindow::onStopClicked()
{
    btn_stop->setEnabled(false);
    btn_stop->setText("停止中...");
    statusLabel->setText("正在强制清理后台...");
    qApp->processEvents();

    // 1. 杀父进程
    if (launch_process->state() == QProcess::Running) {
        launch_process->kill(); 
    }
    
    // 2. 精确查杀所有相关节点
    QString cleanupCmd = 
        // 杀掉检测节点 (Python)
        "pkill -9 -f detect_yolov5; "
        // 杀掉坐标转换节点 (C++)
        "pkill -9 -f transform_node; "
        // 杀掉图像发送节点
        "pkill -9 -f image_viewer; "
        // 杀掉 Realsense
        "pkill -9 -f realsense; "
        // 杀掉组件容器
        "pkill -9 -f component_container; "
        // 杀掉 Launch 脚本
        "pkill -9 -f system_launch.py; " 
        
        // 清理内存
        "rm -f /dev/shm/fastrtps_*; "
        
        // 重启 daemon (为了下一次启动正常)
        "ros2 daemon stop; "
        "ros2 daemon start"; 

    QProcess::execute("bash", QStringList() << "-c" << cleanupCmd);
    
    btn_start->setEnabled(true);
    btn_start->setText("启动系统");
    btn_stop->setText("停止系统");
    le_model_path->setEnabled(true);
    btn_browse->setEnabled(true);
    statusLabel->setText("系统已停止");
}

void MainWindow::onParamChanged(const QString &name, double value) {
    if(ros_worker) {
        ros_worker->setParam(name.toStdString(), value);
    }
}

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
    btn_browse->setMaximumWidth(60); 
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

    // 2. 启动/停止/开关
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

    btn_start = new QPushButton("启 动 系 统", this);
    btn_start->setMinimumWidth(100); 
    btn_start->setStyleSheet("background-color: green; color: white; font-weight: bold;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);
    layActions->addWidget(btn_start);

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
        spinBox->setRange(-2.0, 2.0); // 范围稍微大一点
        spinBox->setSingleStep(0.01);
        spinBox->setValue(defaultVal);
        spinBox->setSuffix(" m"); 
        spinBox->setFixedWidth(100); 

        row->addWidget(spinBox);
        layParam->addLayout(row);

        connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            [this, paramName](double val){
                this->onParamChanged(paramName, val);
            });
    };

    createRow("X Offset (左/右):", "x_offset", 0.00, spin_x);
    createRow("Y Offset (上/下):", "y_offset", 0.00, spin_y);
    createRow("Z Offset (前/后):", "z_offset", 0.00, spin_z);

    mainLayout->addWidget(grpParam);
}