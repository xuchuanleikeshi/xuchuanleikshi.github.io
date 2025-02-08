#密封钉缺陷检测
##本算法针对密封钉缺陷检测，将任务分解为焊道分割和焊道发黑检测两部分，利用多线程技术来执行YOLOv5和U-Net模型的训练与预测，最后用Qt进行界面可视化。

编程环境：
```bash
# 创建虚拟环境（假设使用 Python 3.x）
python -m venv weld_seam_env
# 激活虚拟环境
# Windows
weld_seam_env\Scripts\activate
```
安装所需库
```bash
# 安装 PyQt5（用于界面开发）
pip install PyQt5
# 安装 OpenCV（用于图像处理）
pip install opencv-python
# 安装 PyTorch（用于深度学习模型）
# 根据你的系统和 CUDA 版本选择合适的安装命令
# CPU 版本
pip install torch torchvision torchaudio
# CUDA 11.8 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 安装 NumPy（用于科学计算）
pip install numpy
# 安装其他可能需要的库
pip install glob2  # 用于文件路径匹配
pip install tqdm   # 用于进度条显示（如果代码中有用到）

# 安装 YOLOv5 依赖
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt

# 安装 U-Net 相关依赖（如果有额外的 requirements.txt 文件）
# 假设你的 U-Net 代码在 QtDesigner/model/unet_model.py 中
# 如果有 requirements.txt，可以运行：
pip install -r QtDesigner/model/requirements.txt
```
验证安装：
```bash
# 检查 PyQt5 是否安装成功
python -c "from PyQt5 import QtWidgets; print('PyQt5 installed successfully')"
# 检查 OpenCV 是否安装成功
python -c "import cv2; print(cv2.__version__)"
# 检查 PyTorch 是否安装成功
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

##任务要求如下：
![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E6%A3%80%E6%B5%8B%E8%A6%81%E6%B1%82.JPG)<br>
##数据集预览：
![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-08%20102716.png)
##在焊道分割任务中，U-Net模型通过图像分割技术识别焊道中的发黑区域。U-Net采用对称的编码器-解码器结构，编码器通过卷积和下采样 提取全局特征，解码器通过上采样还原图像细节，生成焊道发黑区域的掩码图。训练过程使用BCEWithLogitsLoss作为损失函数，RMSprop优化器优化模型。数据加载使用ISBI_Loader，并支持GPU加速，以提升训练效率。实践证明该算法模型适用于该焊道检测中焊道分割的精准提取。
##在焊道发黑检测任务中，YOLOv5通过实时目标检测定位焊道发黑缺陷。其骨干网络由卷积层和C3模块组成，结合SPPF模块增强特征提取，并通过FPN和PAN实现多尺度特征融合。模型在P3、P4、P5三个尺度上输出发黑检测结果，使用Depth_Multiple和Width_Multiple参数确保轻量化设计，适用于该工业目标检测场景。
该算法通过结合U-Net的分割能力和YOLOv5的实时目标检测，实现了较为高效与精准的密封钉焊道缺陷检测任务。
##最后的系统可视化界面效果：
![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-06%20120059.png)
项目打包成.exe文件：
![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-08%20105921.png)
.exe程序预览，环境检测通过：<br>
![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-08%20111924.png)
识别结果：<br>
![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-08%20112000.png)
