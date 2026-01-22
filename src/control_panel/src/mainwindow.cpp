#include "control_panel/mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , launch_process(nullptr)
    , ros_worker(nullptr)
{
    // 1. 初始化 UI
    setupUi();

    // 2. 初始化状态栏
    statusBar = new QStatusBar(this);
    setStatusBar(statusBar);
    statusLabel = new QLabel("系统就绪");
    statusBar->addWidget(statusLabel);

    // 3. 初始化 Process
    launch_process = new QProcess(this);
    connect(launch_process, &QProcess::readyReadStandardOutput, [this](){
        QByteArray data = launch_process->readAllStandardOutput();
        qDebug().noquote() << data; 
    });

    setWindowTitle("Projection control panel");
    resize(300, 300); // 保持原来的尺寸
}

MainWindow::~MainWindow() {
    // 1. 杀主进程
    if(launch_process && launch_process->state() == QProcess::Running) {
        launch_process->kill(); 
    }

    // 2. 停 Worker
    destroyWorker();

    if(rclcpp::ok()) {
        rclcpp::shutdown();
    }
    
    forceCleanUp(false); 
}

QString MainWindow::getWorkspacePath(const std::string& subdir) {
    try {
        // 1. 获取当前包 (control_panel) 的 share 路径
        std::filesystem::path p(ament_index_cpp::get_package_share_directory("control_panel"));

        // 2. 向上查找直到找到 'install' 目录
        // 逻辑：share -> install
        while (p.has_parent_path() && p.filename() != "install") {
            p = p.parent_path();
        }

        // 3. 如果找到了 install，它的上一级就是 workspace
        if (p.filename() == "install") {
            std::filesystem::path ws_root = p.parent_path();
            std::filesystem::path target_path = ws_root / subdir; // 拼接 models 或 configs
            
            return QString::fromStdString(target_path.string());
        }
    } catch (const std::exception& e) {
        qWarning() << "Error finding workspace path:" << e.what();
    }

    // 4. 兜底：如果找不到，默认返回 Home 目录
    return QDir::homePath();
}

void MainWindow::destroyWorker() {
    if(ros_worker) {
        // 告诉 ROS 停止 spin
        ros_worker->stop(); 
        
        // 停止 QThread
        ros_worker->quit();
        if(!ros_worker->wait(200)) { // 等待退出，超时强杀
            ros_worker->terminate();
        }
        
        // 删除对象指针
        delete ros_worker;
        ros_worker = nullptr;
    }
}

void MainWindow::forceCleanUp(bool is_blocking) {
    QString cmd = 
        "ps aux | grep -E '"
        // 1. 匹配 Launch 脚本
        "system_launch.py|"
        // 2. 匹配 Realsense 相机二进制文件
        "realsense2_camera_node|"
        // 3. 匹配 YOLO Detect (Python脚本路径包含此名字，精准匹配)
        "yolo_detect_node|"
        // 4. 匹配 YOLO Segment 二进制
        "yolo_seg_node|"
        // 5. 匹配 坐标转换节点
        "transform_node|"
        // 6. 匹配 图像显示节点
        "image_viewer_talker|"
        // 7. 匹配 ROS2 守护进程
        "ros2-daemon"
        "' "
        "| grep -v grep "
        "| grep -v control_panel "
        "| grep -v vscode "
        "| awk '{print $2}' "
        "| xargs -r kill -9";

    // 补充清理：窗口和共享内存
    QString extraCmd = 
        "; "
        "wmctrl -F -c 'Segment Result'; " 
        "wmctrl -F -c 'Detection Result'; "
        "wmctrl -F -c 'projection Image'; "
        "rm -f /dev/shm/fastrtps_*; "
        "rm -f /dev/shm/sem.fastrtps_*; ";

    QString fullCmd = "(" + cmd + extraCmd + ") > /dev/null 2>&1";
    
    if (is_blocking) {
        std::system(fullCmd.toStdString().c_str());
    } else {
        QProcess::startDetached("bash", QStringList() << "-c" << fullCmd);
    }
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
    
    // 强制刷新 UI
    qApp->processEvents();

    if (ros_worker) {
        destroyWorker(); // 防御性编程，确保旧的没了
    }
    ros_worker = new RosWorker(this);
    ros_worker->start();

    // 2. 获取参数
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";
    QString x_val = QString::number(spin_x->value(), 'f', 2);
    QString y_val = QString::number(spin_y->value(), 'f', 2);
    QString z_val = QString::number(spin_z->value(), 'f', 2);
    QString model_str = le_model_path->text().trimmed();
    QString class_str = le_class_path->text().trimmed();

    // 3. 构造脚本
    QString script = QString("ros2 launch cpp_launch system_launch.py " 
                             "show_image:=%1 "
                             "x_offset:=%2 "
                             "y_offset:=%3 "
                             "z_offset:=%4 ")
                             .arg(show_img_val, x_val, y_val, z_val);

    if (!model_str.isEmpty()) {
        script += QString("model_filename:='%1' ").arg(model_str);
    }
    if (!class_str.isEmpty()) {
        script += QString("class_labels_file:='%1' ").arg(class_str);
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
        le_class_path->setEnabled(false);
        btn_browse_class->setEnabled(false);

        statusLabel->setText("系统运行正常");

        if(ros_worker) {
            // 给一点点时间让节点初始化，然后同步参数
            QTimer::singleShot(2000, this, [this](){
               if(ros_worker) ros_worker->setParam("show_image", chk_show_image->isChecked() ? 1.0 : 0.0);
            });
        }

    } else {
        QMessageBox::critical(this, "启动失败", "无法启动 Bash 进程！");
        destroyWorker(); // 启动失败也要清理
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
    qApp->processEvents();

    // 2. 终止 Launch 主进程
    if (launch_process->state() == QProcess::Running) {
        launch_process->kill(); 
    }
    
    // 3. 调用封装好的清理函数
    // 包含了对所有节点的强杀和内存清理
    forceCleanUp(true); 
    destroyWorker();
    // 4. 恢复界面状态
    btn_start->setEnabled(true);
    btn_start->setText("启动系统");
    
    btn_stop->setText("停 止 系 统");
    btn_stop->setEnabled(false);
    
    le_model_path->setEnabled(true);
    btn_browse->setEnabled(true);
    le_class_path->setEnabled(true);
    btn_browse_class->setEnabled(true);

    statusLabel->setText("系统已停止");

    qDebug() << "系统清理完毕"; 
}

void MainWindow::onParamChanged(const QString &name, double value) {
    if(ros_worker) {
        ros_worker->setParam(name.toStdString(), value);
    }
}

// 保持 SetupUI 完全不变
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
    le_model_path->setPlaceholderText("默认:yolov5x_tag_v7.0_detect_640x640_bayese_nv12.bin");
    le_model_path->setReadOnly(true); 
    le_model_path->setToolTip("点击“浏览”选择文件");
    layModel->addWidget(le_model_path);

    btn_browse = new QPushButton("浏览...");
    btn_browse->setMaximumWidth(50); 
    connect(btn_browse, &QPushButton::clicked, [this](){
        QString defaultDir = getWorkspacePath("models");
        QString fileName = QFileDialog::getOpenFileName(
            this, "选择模型文件", defaultDir, "Model Files (*.bin *.onnx);;All Files (*)"
        );
        if (!fileName.isEmpty()) {
            le_model_path->setText(fileName);
        }
    });
    layModel->addWidget(btn_browse);
    layLaunch->addLayout(layModel);

    QHBoxLayout *layClass = new QHBoxLayout();
    layClass->addWidget(new QLabel("类别文件:"));

    le_class_path = new QLineEdit();
    le_class_path->setPlaceholderText("默认:COCO 80类(仅CPU推理支持更换类别文件)");
    le_class_path->setReadOnly(true); 
    le_class_path->setToolTip("选择 .names 或 .txt 文件，为空则使用默认 COCO");
    layClass->addWidget(le_class_path);

    btn_browse_class = new QPushButton("浏览...");
    btn_browse_class->setMaximumWidth(50);
    connect(btn_browse_class, &QPushButton::clicked, [this](){
        QString defaultDir = getWorkspacePath("configs");
        QString fileName = QFileDialog::getOpenFileName(
            this, "选择类别文件", defaultDir, 
            "Class Files (*.names *.txt);;All (*)"
        );
        if (!fileName.isEmpty()) {
            le_class_path->setText(fileName);
        }
    });
    layClass->addWidget(btn_browse_class);
    layLaunch->addLayout(layClass);

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

    // 启动按钮
    btn_start = new QPushButton("启 动 系 统", this);
    btn_start->setMinimumWidth(100); 
    btn_start->setStyleSheet("background-color: green; color: white; font-weight: bold;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);
    layActions->addWidget(btn_start);

    // 停止按钮
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

    createRow("X Offset (正即相机在投影仪右):", "x_offset", 0.01, spin_x);
    createRow("Y Offset (正即相机在投影仪下):", "y_offset", -0.06, spin_y);
    createRow("Z Offset (正即相机在投影仪前):", "z_offset", -0.05, spin_z);

    mainLayout->addWidget(grpParam);
}