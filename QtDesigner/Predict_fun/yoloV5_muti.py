

def process_images(textBrowser,image_folder,model):

    import torch
    import cv2
    import os
    # # 加载预训练好的YOLOv5模型
    # model = torch.hub.load('ultralytics/yolov5', 'custom',
    #                          path='G:\\Weld Seam Defect Detection\\QtDesigner\\weight\\best.pt')
    # model.eval()

    # 初始化计数器
    total_images = 0
    black_defect_count = 0

    # 清空文本浏览器
    textBrowser.clear()
    textBrowser.setPlainText("样本预测结果：\n")  # 设置初始文本

    # 遍历文件夹中的所有图片
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        if image_file.endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
            total_images += 1

            # 读取图片
            image = cv2.imread(image_path)

            print(f"正在处理第 {total_images} 张图片")
            # 进行预测
            results = model(image)

            # 获取预测结果
            predictions = results.xyxy[0]  # 获取所有的框 (x1, y1, x2, y2, confidence, class)


            # 判断是否为缺陷或良品
            if len(predictions) == 0:
                # 如果没有任何标签预测结果，视为良品（Good Product）
                result_text = f"{image_file}: Good Product (no defects detected)"
                # textBrowser.append(result_text)  # 在文本浏览器中追加文本

            else:
                # 检查预测标签
                detected_black_defect = False
                for prediction in predictions:
                    label_index = int(prediction[5])  # 获取预测的标签（class）
                    label_name = results.names[label_index]  # 获取对应的标签名称

                    if label_name == 'Black Defect':
                        # 标签为Black Defect
                        black_defect_count += 1
                        result_text = f"{image_file}: Black Defect detected"
                        # textBrowser.append(result_text)  # 在文本浏览器中追加文本

                        detected_black_defect = True
                        break  # 一旦检测到缺陷，不再继续处理其他框

                if not detected_black_defect:
                    # 如果没有检测到Black Defect，且有其他预测，视为良品
                    result_text = f"{image_file}: Good Product (no defects detected)"
                    # textBrowser.append(result_text)  # 在文本浏览器中追加文本



    # 统计结果
    good_product_count = total_images - black_defect_count
    defect_percentage = (black_defect_count / total_images) * 100 if total_images > 0 else 0

    # 打印统计结果
    textBrowser.append(f"Total Images: {total_images}")
    textBrowser.append(f"Black Defect Count: {black_defect_count}")
    textBrowser.append(f"Good Product Count: {good_product_count}")
    textBrowser.append(f"Defect Percentage: {defect_percentage:.2f}%")
