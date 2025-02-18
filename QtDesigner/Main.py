# -*- coding: utf-8 -*-
import atexit
import glob
import os
import queue

# Form implementation generated from reading ui file 'Weld_Seam_Defect_Detection.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import sys
import threading
import time
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QThread, QObject
import cv2
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap


from QtDesigner.Predict_fun.Predict_Muti_Images import process_images_with_unet
from QtDesigner.Predict_fun.Predict_Single_Image import predict_image
from QtDesigner.Predict_fun.predict_yoloV5 import predict_and_show
from QtDesigner.Predict_fun.yolov5_thread import process_images
from QtDesigner.UI.Weld_Seam_Defect_Detection import Ui_Form
from QtDesigner.model.unet_model import UNet



# 主程序类
# MainWindow 继承自 QWidget，代表主窗口。
class MainWindow(QtWidgets.QWidget):
    # 定义一个信号
    update_message_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.directory = 'D:/OUTPUT/'  # 保存路径
        self.image_folder = 'D:/INPUT/'  # 保存图片文件夹路径
        try:
            # 检查默认输入图片文件夹路径是否存在
            if not os.path.exists(self.image_folder):
                print( f"路径错误默认输入文件夹 {self.image_folder} 不存在，请选择有效的文件夹。")
                self.image_folder = QFileDialog.getExistingDirectory(self, "默认输入文件夹不存在，请选择输入图片文件夹", "",
                                                                     QFileDialog.ShowDirsOnly)
            else:
                self.image_folder = self.image_folder  # 使用默认路径

            # 检查默认输出文件夹路径是否存在
            if not os.path.exists(self.directory):
                print(f"默认输出文件夹 {self.directory} 不存在，请选择有效的文件夹。")
                self.directory = QFileDialog.getExistingDirectory(self, "默认输出文件夹不存在，请选择输出文件夹", "", QFileDialog.ShowDirsOnly)
            else:
                self.directory = self.directory  # 使用默认路径
        except Exception as e:
            print(f"发生错误: {e}\n请检查文件路径设置")
            return  # 或者可以选择继续执行其他逻辑

        self.image_files = []  # 保存文件夹内的图片文件名
        self.current_index = 0  # 当前显示的图片索引
        self.len_folder = 0

        try:
            self.image_files = [f for f in os.listdir(self.image_folder) if
                                f.lower().endswith(('.png', '.jpg', '.bmp'))]  # 获取所有图片文件
            self.image_files.sort()  # 按字母排序文件名
            self.len_folder = len(self.image_files)
            if self.image_files:  # 如果文件夹内有图片
                self.current_index = 0  # 重置当前索引
                # self.show_image()  # 显示第一张图片
            # self.image_files = []  # 保存文件夹内的图片文件名
            # self.current_index = 0  # 当前显示的图片索引

            # 连接信号到槽
            self.update_message_signal.connect(self.update_text_edit)
        except Exception as e:
            print(f"读取图片文件时发生错误: {e}\n请检查文件夹内容。")
            return  # 或者可以选择继续执行其他逻辑

        self.message =[]
        self.idx =0
        self.ids =0
        # self.len_folder =0
        # 获取模型路径
        if hasattr(sys, '_MEIPASS'):
            self.model_path = os.path.join(sys._MEIPASS, 'weight/best_model_test_150.pth')
        else:
            self.model_path = './weight/best_model_test_150.pth'

        # 设置 TORCH_HOME 环境变量为相对路径
        torch_cache_dir = os.path.join(os.path.dirname(__file__), 'torch_cache')  # 使用当前文件所在目录作为基准
        os.environ['TORCH_HOME'] = torch_cache_dir  # 将缓存目录设置为相对路径

        # yolov5预加载
        if hasattr(sys, '_MEIPASS'):
            self.yolo_path = os.path.join(sys._MEIPASS, 'weight/best.pt')
        else:
            self.yolo_path = './weight/best.pt'
        # 加载预训练模型 (假设权重文件位于当前目录下，且为 best_model.pt)
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom',
        #                        path=str(Path("G:\\Weld_Seam_Defect_Detection\\QtDesigner\\weight\\best.pt").resolve()))
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.yolo_path)
        # 设置模型为 eval 模式（推理模式）
        self.model.eval()

        # unet预加载
        # self.model_path = './weight/best_model_test_150.pth'
        # 加载模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1
        self.net = UNet(n_channels=1, n_classes=1)
        self.net.to(device=self.device)
        # 加载模型参数
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net.eval()  # 切换到评估模式



    # 定义打开图片文件夹事件函数
    def open_folder(self):
        print("图片文件夹事件已响应")
        # 弹出文件夹选择对话框
        folder = QFileDialog.getExistingDirectory(self, "打开图片文件夹")

        if folder:  # 如果选择了文件夹
            self.image_folder = folder
            self.image_files = [f for f in os.listdir(folder) if
                                f.lower().endswith(('.png', '.jpg', '.bmp'))]  # 获取所有图片文件
            self.image_files.sort()  # 按字母排序文件名
            self.len_folder = len(self.image_files)
            if self.image_files:  # 如果文件夹内有图片
                self.current_index = 0  # 重置当前索引
                self.show_image()  # 显示第一张图片


    # 显示当前索引的图片
    def show_image(self):
        if self.image_files:  # 如果有图片
            file_path = os.path.join(self.image_folder, self.image_files[self.current_index])  # 获取图片路径
            pixmap = QPixmap(file_path)  # 加载图片

            # 调整标签的大小以适应图片的比例
            aspect_ratio = pixmap.width() / pixmap.height()  # 计算宽高比
            label_width = self.ui.Origin_Image.width()
            label_height = self.ui.Origin_Image.height()

            if aspect_ratio > 1:  # 横向图片
                new_height = label_height
                new_width = int(new_height * aspect_ratio)
            else:  # 纵向图片
                new_width = label_width
                new_height = int(new_width / aspect_ratio)

            # 设置 QLabel 大小
            self.ui.Origin_Image.setFixedSize(new_width, new_height)

            # 将图片显示在 Origin_Image 标签上
            self.ui.Origin_Image.setPixmap(pixmap.scaled(new_width, new_height, QtCore.Qt.KeepAspectRatio))

    # 定义下一张图片事件函数
    def next_image(self):
        print("下一张图片事件已响应")
        # 改变图片索引current_index，实现图片的下一张，并调用show_image，将图片显示在图片显示在 Origin_Image 标签上
        if self.image_files:  # 如果有图片
            self.current_index = (self.current_index + 1) % len(self.image_files)  # 更新索引，循环回到第一张
            self.show_image()  # 显示下一张图片


    # 定义上一张图片事件函数
    def prev_image(self):
        print("上一张图片事件已响应")
        # 改变图片索引current_index，实现图片的上一张，并调用show_image，将图片显示在图片显示在 Origin_Image 标签上
        if self.image_files:  # 如果有图片
            self.current_index = (self.current_index - 1) % len(self.image_files)  # 更新索引，循环回到第一张
            self.show_image()  # 显示上一张图片

    # 定义检测图片函数
    def detect_image(self):
        print("检测图片事件已响应")
        try:
            # 获取图片路径
            file_path = os.path.join(self.image_folder, self.image_files[self.current_index])  # 获取图片路径
            # 先处理焊道分割问题
            pre_image = predict_image(file_path,self.net ,self.device,model_path='../weight/best_model_test_150.pth')
            # 显示预测结果
            self.show_weld_seam_image(pre_image)

            # 实现图像黑点检测
            black_image = predict_and_show(file_path,self.model)
            # 显示预测结果
            self.show_Black_Detect_image(black_image)

            # 显示原图像
            self.show_image()
        except Exception as e:
            print(f"{e}")


    def show_weld_seam_image(self, image):
        """将焊道分割结果显示在 Weld_Seam 标签上"""
        # 将图像转换为 QImage
        height, width = image.shape[:2]
        bytes_per_line = 3 * width  # 每行字节数（假设是RGB图像）
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # 创建 QPixmap 并设置到 QLabel
        pixmap = QtGui.QPixmap.fromImage(q_image)

        # 调整标签的大小以适应图片的比例
        aspect_ratio = pixmap.width() / pixmap.height()  # 计算宽高比
        label_width = self.ui.Weld_Seam.width()
        label_height = self.ui.Weld_Seam.height()

        if aspect_ratio > 1:  # 横向图片
            new_height = label_height
            new_width = int(new_height * aspect_ratio)
        else:  # 纵向图片
            new_width = label_width
            new_height = int(new_width / aspect_ratio)

        # 设置 QLabel 大小
        self.ui.Weld_Seam.setFixedSize(new_width, new_height)

        # 将图片显示在 Origin_Image 标签上
        self.ui.Weld_Seam.setPixmap(pixmap.scaled(new_width, new_height, QtCore.Qt.KeepAspectRatio))

    def show_Black_Detect_image(self, image):
        """将黑点检测结果显示在 Weld_Seam 标签上"""
        # 将图像转换为 QImage
        height, width = image.shape[:2]
        bytes_per_line = 3 * width  # 每行字节数（假设是RGB图像）
        q_image = QtGui.QImage(image.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)

        # 创建 QPixmap 并设置到 QLabel
        pixmap = QtGui.QPixmap.fromImage(q_image)

        # 调整标签的大小以适应图片的比例
        aspect_ratio = pixmap.width() / pixmap.height()  # 计算宽高比
        label_width = self.ui.Black_Detect.width()
        label_height = self.ui.Black_Detect.height()

        if aspect_ratio > 1:  # 横向图片
            new_height = label_height
            new_width = int(new_height * aspect_ratio)
        else:  # 纵向图片
            new_width = label_width
            new_height = int(new_width / aspect_ratio)

        # 设置 QLabel 大小
        self.ui.Black_Detect.setFixedSize(new_width, new_height)

        # 将图片显示在 Origin_Image 标签上
        self.ui.Black_Detect.setPixmap(pixmap.scaled(new_width, new_height, QtCore.Qt.KeepAspectRatio))

    def save_path(self):
         """打开文件夹对话框以选择保存文件夹"""
         self.directory = QFileDialog.getExistingDirectory(None, "选择保存文件夹", "")
         # 检查用户是否选择了路径，若没有选择返回 None
         return self.directory if self.directory else None

    # 批量处理图片
    # 批量处理图片
    def test_Set(self):
        try:
            self.ui.progressBar.setMaximum(self.len_folder)
            self.ui.progressBar.setValue(self.idx)
            self.ui.progressBar_2.setMaximum(self.len_folder)
            self.ui.progressBar_2.setValue(self.ids)


            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_bar)  # 定时器定期触发任务
            self.timer.start(100)  # 每100ms触发一次

            # 创建线程
            thread = threading.Thread(target=self.perll)
            thread_1= threading.Thread(target=self.perrl)
            # 启动线程
            thread.start()
            thread_1.start()
            # 等待线程结束
            thread.join()
            thread_1.join()
        except Exception as e:
            print(f"线程一执行中出现异常: {e}")
        finally:
            print("线程一执行完毕")

    def perll(self):

        thread = threading.Thread(target=process_images,
                                  args=(self.image_folder, self.model, self.directory, self.update_progress,self.up_text))
        thread.start()

    def update_progress(self,idx):
        self.idx =idx
        # print(f"当前处理黑点的图片序号{self.idx}")

    def up_text(self,message):
        self.message.append(message)
        # print(f"预测数据为{self.message}")
        self.update_message_signal.emit(message)  # 发射信号

    def update_text_edit(self,message):
        self.ui.textEdit.append(message)# 更新 UI

    def update_bar(self):
        self.ui.progressBar.setValue(self.idx)
        self.ui.progressBar_2.setValue(self.ids)

        if (self.ids == self.len_folder and self.idx == self.len_folder):
            self.qmessageBox()
            self.timer.stop()

    # 分割线程
    def perrl(self):

        thread = threading.Thread(target=process_images_with_unet,
                                  args=(glob.glob(os.path.join(self.image_folder, '*.jpg')), self.directory, self.net, self.device,self.upbar))
        thread.start()
    def upbar(self,idx):
        self.ids =idx
        print(f"当前处理分割的图片序号{self.ids}")
    def qmessageBox(self):
        """显示保存成功的提示框"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("保存完成")
        msg_box.setText(f"文件已成功保存！！！,保存路径为: {self.directory}/10617 重庆邮电大学 徐传磊.txt")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

# 主程序入口
if __name__ == '__main__':
    try:
        app = QtWidgets.QApplication(sys.argv)  # 创建应用程序对象

        # 创建主窗口对象
        main_window = MainWindow()

        # 显示窗口
        main_window.show()

        # 进入事件循环
        sys.exit(app.exec_())

    except Exception as e:
        print(f"主线程执行中出现异常: {e}")

    finally:
        print("主线程执行完毕")
        os._exit(1)



