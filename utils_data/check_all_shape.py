import json
import numpy as np
import os

def check_npy_images_in_json(json_path):
    # 读取 JSON 文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 遍历 JSON 数据中的每一项
    for idx, item in enumerate(data):
        # 假设 JSON 文件中每个项包含 'image' 键指向 .npy 文件路径，可以是单个文件路径或路径列表
        image_paths = item.get('image')  # 获取 image 字段

        # 如果 image_paths 是列表，则遍历列表中的每个路径
        if isinstance(image_paths, list):
            for image_path in image_paths:
                check_npy_image(image_path, idx)
        elif isinstance(image_paths, str):  # 如果是单个路径
            check_npy_image(image_paths, idx)
        else:
            print(f"Warning: No valid image path for entry {idx} in JSON")

def check_npy_image(image_path, idx):
    # 检查 image_path 是否为有效路径
    if image_path and isinstance(image_path, str) and os.path.exists(image_path):
        try:
            # 加载图像数据
            image = np.load(image_path)
            
            # 检查图像是否是 (1, 32, 256, 256)
            if image.shape != (1, 32, 256, 256):
                print(f"Warning: Image {idx} at {image_path} does not have the expected shape (1, 32, 256, 256), instead it has shape {image.shape}")
        except Exception as e:
            print(f"Error loading image {idx} at {image_path}: {e}")
    else:
        print(f"Warning: Invalid path for image {idx} at {image_path}")

if __name__ == "__main__":
    # 输入你的 JSON 文件路径
    json_file_path = "/home/zwding/ly/json_data/train/train_3d_5.json"
    check_npy_images_in_json(json_file_path)
