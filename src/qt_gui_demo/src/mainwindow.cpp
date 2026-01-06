#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QRandomGenerator>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_talkerButton_clicked()
{
    talkerProcess = new QProcess(this);
    talkerProcess->start("bash", QStringList() << "-c" << "source /opt/ros/humble/setup.bash && source /home/sunrise/projection_ws/install/setup.bash && ros2 run image_viewer image_viewer_talker");
}

void MainWindow::on_listenerButton_clicked()
{
    listenerProcess = new QProcess(this);
    listenerProcess->start("bash", QStringList() << "-c" << "source /opt/ros/humble/setup.bash && source /home/sunrise/projection_ws/install/setup.bash && ros2 run image_viewer image_viewer_listener");
}

void MainWindow::on_publisherButton_clicked()
{
    // 生成随机数量的圆（1~10个）
    int count = QRandomGenerator::global()->bounded(1, 11);

    // 构造 CircleArray 消息字符串
    QStringList circleList;
    for (int i = 0; i < count; ++i) {
        int x = QRandomGenerator::global()->bounded(0, 1921);
        int y = QRandomGenerator::global()->bounded(0, 1081);
        int radius = QRandomGenerator::global()->bounded(0, 101);
        int color = QRandomGenerator::global()->bounded(0, 2);

        QString circleStr = QString("{x: %1, y: %2, radius: %3, color: %4}")
                            .arg(x).arg(y).arg(radius).arg(color);
        circleList << circleStr;
    }

    QString circlesStr = "circles: [" + circleList.join(", ") + "]";

    // 启动 ROS2 发布进程
    QString command = QString("source /opt/ros/humble/setup.bash && "
                               "source /home/sunrise/projection_ws/install/setup.bash && "
                               "ros2 topic pub /circle_param custom_msgs/msg/CircleArray \"%1\"")
                               .arg(circlesStr);

    publisherProcess = new QProcess(this);
    publisherProcess->start("bash", QStringList() << "-c" << command);
}
