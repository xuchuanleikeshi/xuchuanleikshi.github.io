import glob
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import cv2
import os
from QtDesigner.model.unet_model import UNet


def process_images_with_unet(test_paths, save_dir, net, device,upbar):
    """
      使用U-Net模型对图片进行预测，并保存结果和叠加图片

      :param test_paths: 需要预测的图片路径列表
      :param save_dir: 结果保存的文件夹路径
      :param model_path: U-Net模型的路径
      """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    idx =0
    def process_single_image(test_path):
        """
        单独处理一张图片并返回结果路径

        :param test_path: 图片路径
        :return: 处理结果路径
        """
        # 读取图片并转为灰度图
        img = cv2.imread(test_path)
        if img is None:
            print(f"Error reading image {test_path}. Skipping...")
            return None, None

        origin_shape = img.shape
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize到512x512
        img_resized = cv2.resize(img_gray, (512, 512))
        img_tensor = torch.from_numpy(img_resized.reshape(1, 1, 512, 512)).to(device=device, dtype=torch.float32)

        # 预测
        with torch.no_grad():  # 禁用梯度计算
            pred = net(img_tensor)

        # 处理结果
        pred = np.array(pred.data.cpu()[0])[0]
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # Resize回原始形状
        pred_resized = cv2.resize(pred.astype(np.uint8), (origin_shape[1], origin_shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

        # 生成保存路径
        file_name = os.path.basename(test_path).split('.')[0]
        save_res_path = os.path.join(save_dir, f"{file_name}_res.png")
        save_overlay_path = os.path.join(save_dir, f"{file_name}_overlay.png")

        # 创建叠加层
        overlay = img.copy()
        alpha = 0.6  # 叠加透明度
        mask_color = [0, 0, 255]  # 红色作为mask的颜色

        # 使用掩码将 `pred_resized` 为255的区域染成红色
        overlay[pred_resized == 255] = mask_color
        result_overlay = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # 保存结果图片
        cv2.imwrite(save_res_path, pred_resized)
        cv2.imwrite(save_overlay_path, result_overlay)

        return save_res_path, save_overlay_path

    # 使用线程池并行处理图片
    with ThreadPoolExecutor(max_workers=4)  as executor:
        futures = {executor.submit(process_single_image, test_path): test_path for test_path in test_paths}

        for future in futures:
            try:
                idx +=1
                upbar(idx)
                res_path, overlay_path = future.result()
                if res_path and overlay_path:
                    print(f"Processed: {res_path}, Overlay: {overlay_path}")
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")


if __name__ == '__main__':

    # 使用 glob 获取所有待预测图片路径
    # test_paths = glob.glob('G:/INPUT/*.jpg')
    input_dir ='G:/INPUT'
    test_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    save_dir='G:/OUTPUT'

    model_path = '../weight/best_model_test_150.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(n_channels=1, n_classes=1)
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 切换到评估模式
    process_images_with_unet(test_paths, save_dir, net, device)
