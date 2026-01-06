#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QProcess>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_talkerButton_clicked();
    void on_listenerButton_clicked();
    void on_publisherButton_clicked();

private:
    Ui::MainWindow *ui;
    QProcess *talkerProcess;
    QProcess *listenerProcess;
    QProcess *publisherProcess;
};

#endif // MAINWINDOW_H
