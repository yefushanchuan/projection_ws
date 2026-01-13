#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QStatusBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QCloseEvent>
#include <QApplication>
#include <QThread>
#include <QDebug>

#include "control_panel/rosworker.h" 

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent *event) override;

private slots:
    void onStartClicked();
    void onStopClicked();
    void onParamChanged(const QString &name, double value);

private:
    void setupUi();

    // UI 指针
    QLineEdit *le_model_path;
    QPushButton *btn_browse;
    QPushButton *btn_start;
    QPushButton *btn_stop;
    QCheckBox *chk_show_image;
    
    QDoubleSpinBox *spin_x;
    QDoubleSpinBox *spin_y;
    QDoubleSpinBox *spin_z;

    QStatusBar *statusBar;
    QLabel *statusLabel;

    // 后台逻辑指针
    QProcess *launch_process;
    RosWorker *ros_worker;
};

#endif // MAINWINDOW_H