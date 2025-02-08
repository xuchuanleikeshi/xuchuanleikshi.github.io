import os

import torch
import cv2
from matplotlib import pyplot as plt
from pathlib import Path


def predict_and_show(image_path,model):
    # # 加载预训练模型 (假设权重文件位于当前目录下，且为 best_model.pt)
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(Path("G:\\Weld Seam Defect Detection\\QtDesigner\\weight\\best.pt").resolve()))
    #
    # # 设置模型为 eval 模式（推理模式）
    # model.eval()

    # 读取输入图片
    # image_path = 'G:\\traintest\\1 (1).jpg'  # 替换为你的图片路径
    image = cv2.imread(image_path)

    # 对图像进行推理
    results = model(image)

    # 获取预测结果
    predictions = results.xyxy[0]  # 获取所有的框 (x1, y1, x2, y2, confidence, class)

    # 可视化预测结果
    # results.show()  # 使用 YOLOv5 的可视化功能

    # 如果要在 Matplotlib 中显示结果
    results.render()  # 在图像上绘制预测结果
    # plt.figure()
    # plt.imshow(cv2.cvtColor(results.ims[0], cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    # 保存带有预测框的图像

    # 获取当前工作目录并生成保存路径
    # current_directory = os.getcwd()
    # save_path = Path(current_directory) / save_filename

    output_image = results.ims[0]  # 获取带有框的图像

    # 显示图像
    # cv2.imshow("预测结果", cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)  # 按任意键关闭窗口
    # cv2.destroyAllWindows()

    # save_path = Path(save_path)  # 将保存路径转换为 Path 对象
    # cv2.imwrite(str(save_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))  # 保存图像
    #
    # print(f"结果图像已保存至: {save_path}")
    # 返回带有预测框的图像
    print("成功识别黑点")
    return cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)


# 测试函数
if __name__ == "__main__":

    # 预加载
    # 加载预训练模型 (假设权重文件位于当前目录下，且为 best_model.pt)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(Path("G:\\Weld Seam Defect Detection\\QtDesigner\\weight\\best.pt").resolve()))

    # 设置模型为 eval 模式（推理模式）
    model.eval()

    image_path = 'G:\\traintest\\1 (1).jpg'  # 替换为你的图片路径
    figure = predict_and_show(image_path,model)
    plt.figure()
    plt.imshow(cv2.cvtColor(figure, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

