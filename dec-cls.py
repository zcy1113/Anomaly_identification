import os
import glob
from PIL import Image
from ultralytics import YOLO

# 配置路径
model_path = 'best-pc-1115.pt'  # 替换为你的模型路径
input_folder = 'test_image/test'  # 输入图片文件夹
output_folder = 'test_image/test1'  # 输出文件夹
confidence_threshold = 0.5  # 置信度阈值

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 加载模型
model = YOLO(model_path)

# 获取输入文件夹中的所有图片
image_files = glob.glob(os.path.join(input_folder, '*.png'))

# 处理每个图片文件
for image_path in image_files:
    print(f'Processing {image_path}')

    # 加载原始图片
    original_image = Image.open(image_path)

    # 执行推理
    results = model.predict(source=image_path, conf=confidence_threshold,imgsz=1088)

    # 获取检测框、类别名
    boxes = results[0].boxes  # 检测框
    names = results[0].names  # 类别名称映射

    # 提取每个框内的物体并保存
    for i, box in enumerate(boxes):
        # 提取框的坐标 (x_min, y_min, x_max, y_max)
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

        # 提取框的类别名称
        cls_id = int(box.cls[0])  # 类别索引
        class_name = names.get(cls_id, "unknown")  # 类别名称

        # 裁剪框内的物体
        cropped_img = original_image.crop((x_min, y_min, x_max, y_max))

        # 构造输出文件名
        base_name = os.path.basename(image_path)
        file_name, _ = os.path.splitext(base_name)
        output_file = os.path.join(output_folder, f"{file_name}_{class_name}_{i + 1}.png")

        # 保存裁剪后的图片
        cropped_img.save(output_file)
        print(f"Saved cropped object to {output_file}")

print("All images processed and objects saved.")


