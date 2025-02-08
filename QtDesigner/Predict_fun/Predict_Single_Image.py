import numpy as np
import torch
import cv2
from QtDesigner.model.unet_model import UNet


def predict_image(image_path,net,device,model_path='./weight/best_model_test_150.pth'):
    """
    对单张图片进行预测并保存结果。

    参数:
    - image_path: str，输入的图片路径
    - model_path: str，模型的路径
    - device: torch.device, 使用的设备（'cuda' 或 'cpu'），默认为None，自动选择

    返回:
    - save_res_path: str, 预测的mask结果保存路径
    - save_overlay_path: str, 叠加原图的结果保存路径
    """
    # 选择设备

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')
    #
    # # 加载网络，图片单通道，分类为1
    # net = UNet(n_channels=1, n_classes=1)
    # net.to(device=device)

    # # 加载模型参数
    # net.load_state_dict(torch.load(model_path, map_location=device))
    # net.eval()  # 切换到评估模式

    # 保存结果地址
    save_res_path = image_path.split('.')[0] + '_res.png'
    save_overlay_path = image_path.split('.')[0] + '_overlay.png'

    # 读取图片
    img = cv2.imread(image_path)
    origin_shape = img.shape
    print(f"Original shape: {origin_shape}")

    # 转为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (512, 512))

    # 转为batch为1，通道为1，大小为512*512的数组
    img_resized = img_resized.reshape(1, 1, img_resized.shape[0], img_resized.shape[1])

    # 转为tensor
    img_tensor = torch.from_numpy(img_resized)
    img_tensor = img_tensor.to(device=device, dtype=torch.float32)

    # 预测
    with torch.no_grad():  # 禁用梯度计算
        pred = net(img_tensor)

    # 提取结果
    pred = np.array(pred.data.cpu()[0])[0]

    # 处理结果
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0

    # 将单通道的预测结果转为原始尺寸
    pred_resized = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_resized = pred_resized.astype(np.uint8)

    # 创建一个叠加层，给预测结果加颜色
    overlay = img.copy()
    alpha = 0.6  # 叠加透明度
    mask_color = [0, 0, 255]  # 红色作为mask的颜色

    # 使用掩码将 `pred_resized` 为255的区域染成红色
    overlay[pred_resized == 255] = mask_color

    # 将叠加层和原始图像进行加权叠加
    result_overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 保存结果图片
    # cv2.imwrite(save_res_path, pred_resized)
    # cv2.imwrite(save_overlay_path, result_overlay)

    # print(f"Result saved to {save_res_path}")
    # print(f"Overlay saved to {save_overlay_path}")
    print("成功识别焊道")
    return result_overlay


# 示例用法
if __name__ == "__main__":

    image_path = 'G:\\test\\NG_val_0001.jpg'  # 替换为实际图片路径
    model_path = '../weight/best_model_test_150.pth'

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 切换到评估模式

    predict_image(image_path,net ,device,model_path='../weight/best_model_test_150.pth')
