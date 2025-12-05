import json

def limit_image_paths(json_file, output_file):
    """
    读取 JSON 文件并限制 [image] 字段中的图像路径数量最多为 5。

    Args:
        json_file (str): 输入的 JSON 文件路径。
        output_file (str): 输出的 JSON 文件路径。
    """
    try:
        # 读取 JSON 文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 遍历 JSON 数据并限制 [image] 中的图像路径数量
        for block in data:
            if "image" in block and isinstance(block["image"], list):
                block["image"] = block["image"][:3]

        # 写入新的 JSON 文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"处理完成！结果已保存到 {output_file}")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例使用
if __name__ == "__main__":
    input_json = "/home/zwding/ly/LongLLaVA3D/json_data/train_data/train_3d_5-stage-CoT.json"  # 替换为你的输入 JSON 文件路径
    output_json = "/home/zwding/ly/LongLLaVA3D/json_data/train_data/train_3d_3-stage-CoT.json"  # 替换为你想要保存的输出 JSON 文件路径
    limit_image_paths(input_json, output_json)
