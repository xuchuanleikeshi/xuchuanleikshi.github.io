本算法针对密封钉缺陷检测，将任务分解为焊道分割和焊道发黑检测两部分，利用多线程技术来执行YOLOv5和U-Net模型的训练与预测，最后用Qt进行界面可视化。
任务要求如下：![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E6%A3%80%E6%B5%8B%E8%A6%81%E6%B1%82.JPG)
数据集预览：![image](https://github.com/xuchuanleikeshi/xuchuanleikshi.github.io/blob/main/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-02-08%20102716.png)
  在焊道分割任务中，U-Net模型通过图像分割技术识别焊道中的发黑区域。U-Net采用对称的编码器-解码器结构，编码器通过卷积和下采样 提取全局特征，解码器通过上采样还原图像细节，生成焊道发黑区域的掩码图。训练过程使用BCEWithLogitsLoss作为损失函数，RMSprop优化器优化模型。数据加载使用ISBI_Loader，并支持GPU加速，以提升训练效率。实践证明该算法模型适用于该焊道检测中焊道分割的精准提取。
  在焊道发黑检测任务中，YOLOv5通过实时目标检测定位焊道发黑缺陷。其骨干网络由卷积层和C3模块组成，结合SPPF模块增强特征提取，并通过FPN和PAN实现多尺度特征融合。模型在P3、P4、P5三个尺度上输出发黑检测结果，使用Depth_Multiple和Width_Multiple参数确保轻量化设计，适用于该工业目标检测场景。
该算法通过结合U-Net的分割能力和YOLOv5的实时目标检测，实现了较为高效与精准的密封钉焊道缺陷检测任务。
