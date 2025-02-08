import time
from concurrent.futures import ThreadPoolExecutor
import os
import cv2
import torch
from pathlib import Path

def process_images(image_folder, model, log_folder_name,update_progress,uptext):
    try:
        # 定义日志文件夹路径
        log_file_path = Path("logs") / log_folder_name
        log_file_path.mkdir(parents=True, exist_ok=True)  # 创建日志文件夹

        # 获取文件夹中的所有图片文件列表
        # image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 图片顺序读写
        # 列出文件夹中所有图片文件，按文件名排序
        image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

        total_images = len(image_files)
        black_defect_count = 0

        # 设置进度条的最大值为图片总数
        # progressBar.setMaximum(total_images)
        # progressBar.setValue(0)
        # 存储每个文件的结果
        results_list = []

        # 定义消息保存函数
        def save_message_to_file(message, file_path):
            try:
                # 将消息写入日志文件，追加模式
                with open(file_path, 'a', encoding='utf-8') as file:
                    file.write(message + "  ")
                print(f"消息已成功保存到 {file_path}")
            except Exception as e:
                print(f"保存文件时出错: {e}")

        # 定义图片处理函数
        def process_single_image(image_file):
            nonlocal black_defect_count
            image_path = os.path.join(image_folder, image_file)
            print(f"图片处理顺序:{image_path}")
            try:
                # 读取图片
                image = cv2.imread(str(image_path))
                if image is None:
                    error_message = f"无法读取图片: {image_file}"
                    results_list.append(error_message)
                    return

                # 模型推断
                with torch.no_grad():  # 禁用梯度计算以加速推断
                    results = model(image)

                # 使用结果渲染方法绘制预测框
                results.render()  # 在图像上绘制预测结果
                output_image = results.ims[0]  # 获取带有框的图像

                # 保存处理后的图片到指定路径
                processed_image_path = log_file_path / f"{image_file}"
                cv2.imwrite(str(processed_image_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))  # 保存图像

                predictions = results.xyxy[0]  # 获取所有的框 (x1, y1, x2, y2, confidence, class)

                if len(predictions) == 0:
                    result_text = "0"
                    results_list.append(result_text)
                else:
                    detected_black_defect = False
                    for prediction in predictions:
                        label_index = int(prediction[5])
                        label_name = results.names[label_index]

                        if label_name == 'Black Defect':
                            black_defect_count += 1
                            result_text = "1"
                            results_list.append(result_text)
                            detected_black_defect = True
                            break

                    if not detected_black_defect:
                        result_text = "0"
                        results_list.append(result_text)

            except Exception as e:
                error_message = f"处理图片时出错: {image_file}, 错误: {e}"
                # results_list.append(error_message)

        # 使用线程池并行处理图片
        with ThreadPoolExecutor(max_workers=4) as executor:  # 限制线程数量，避免 GPU 资源争用
            futures = [executor.submit(process_single_image, image_file) for image_file in image_files]

            for idx, future in enumerate(futures):
                future.result()  # 等待每个任务完成
                update_progress(idx+1)  # 调用回调函数，传入当前索引


        # 统计结果
        summary_message = (
            f"  \n测试数量：{total_images}  \n"
            f"发黑数量：{black_defect_count}  \n"
            f"良品数：{total_images - black_defect_count}  \n"
            f"缺陷率：{(black_defect_count / total_images) * 100:.2f}%  \n"
        )

        results_list.append(summary_message)
        uptext(summary_message)
        # 将所有结果写入日志文件
        log_file_name = log_file_path / "10617 重庆邮电大学 徐传磊.txt"  # 定义日志文件名
        for message in results_list:
            save_message_to_file(message, log_file_name)

    except Exception as e:
        print(f"图片处理函数异常: {e}")
    finally:
        print("图片处理函数执行完毕")
