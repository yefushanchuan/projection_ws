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
    QGroupBox *grpLaunch = new QGroupBox("系统启动控制", this);
    QVBoxLayout *layLaunch = new QVBoxLayout(grpLaunch);
    
    chk_show_image = new QCheckBox("开启图像显示 (show_image)", this);
    layLaunch->addWidget(chk_show_image);

    QHBoxLayout *layButtons = new QHBoxLayout();
    btn_start = new QPushButton("启动 Launch", this);
    btn_start->setStyleSheet("background-color: green; color: white;");
    connect(btn_start, &QPushButton::clicked, this, &MainWindow::onStartClicked);

    btn_stop = new QPushButton("停止 Launch", this);
    btn_stop->setStyleSheet("background-color: red; color: white;");
    btn_stop->setEnabled(false);
    connect(btn_stop, &QPushButton::clicked, this, &MainWindow::onStopClicked);

    layButtons->addWidget(btn_start);
    layButtons->addWidget(btn_stop);
    layLaunch->addLayout(layButtons);
    mainLayout->addWidget(grpLaunch);

    // --- 参数调节区 ---
    QGroupBox *grpParam = new QGroupBox("实时参数调节", this);
    QVBoxLayout *layParam = new QVBoxLayout(grpParam);

    // 辅助 lambda: 快速创建 Label + SpinBox 的一行
    auto createRow = [&](const QString &label, const QString &paramName, double defaultVal, QDoubleSpinBox* &spinBox) {
        QHBoxLayout *row = new QHBoxLayout();
        row->addWidget(new QLabel(label));
        spinBox = new QDoubleSpinBox();
        spinBox->setRange(-10.0, 10.0);
        spinBox->setSingleStep(0.01);
        spinBox->setValue(defaultVal);
        row->addWidget(spinBox);
        layParam->addLayout(row);

        // 值改变 -> 触发 onParamChanged
        connect(spinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            [this, paramName](double val){
                this->onParamChanged(paramName, val);
            });
    };

    createRow("X Offset:", "x_offset", 0.0, spin_x);
    createRow("Y Offset:", "y_offset", 0.0, spin_y);
    createRow("Z Offset:", "z_offset", 0.0, spin_z);

    mainLayout->addWidget(grpParam);
}

void MainWindow::onStartClicked()
{
    QString show_img_val = chk_show_image->isChecked() ? "true" : "false";
    
    // 构造命令
    // 请确保 source 路径正确
    QString script = "source /opt/ros/humble/setup.bash && "
                     "ros2 launch cpp_launch system_launch.py " 
                     "show_image:=" + show_img_val;

    qDebug() << "Executing:" << script;

    launch_process->start("bash", QStringList() << "-c" << script);

    btn_start->setEnabled(false);
    btn_stop->setEnabled(true);
    chk_show_image->setEnabled(false); // 启动后锁定参数
}

void MainWindow::onStopClicked()
{
    launch_process->terminate();
    btn_start->setEnabled(true);
    btn_stop->setEnabled(false);
    chk_show_image->setEnabled(true);
}

void MainWindow::onParamChanged(const QString &name, double value) {
    // 将 UI 的请求转发给 Worker 线程
    if(ros_worker) {
        ros_worker->setParam(name.toStdString(), value);
    }
}