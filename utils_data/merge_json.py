import json
import os
import uuid

def get_patient_hospital_info(image_path):
    # 提取图片路径中的倒数第三层级和倒数第二层级，分别为医院和病人名字
    path_parts = image_path[0].split(os.sep)
    hospital = path_parts[-4]  # 倒数第三层级
    patient_name = path_parts[-3]  # 倒数第二层级
    return hospital, patient_name

def get_patient_hospital_info_3d(image_path):
    # 提取图片路径中的倒数第三层级和倒数第二层级，分别为医院和病人名字
    path_parts = image_path[0].split(os.sep)
    hospital = path_parts[-3]  # 倒数第三层级
    patient_name = path_parts[-2]  # 倒数第二层级
    return hospital, patient_name


def merge_json_files(json_file_1, json_file_2, output_file):
    # 读取两个 JSON 文件
    with open(json_file_1, 'r', encoding='utf-8') as f1, open(json_file_2, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # 创建一个字典，以医院和病人名字为键，存储对应的图片路径
    merged_data = {}

    # 处理第一个 JSON 文件
    for item in data1:
        image_path = item['image']
        hospital, patient_name = get_patient_hospital_info(image_path)
        key = (hospital, patient_name)
        if key not in merged_data:
            merged_data[key] = {'image_2d': [], 'image_3d': [], 'conversations_2d': item.get('conversations', []), 'conversations_3d': item.get('conversations', [])}
        merged_data[key]['image_2d'].append(image_path)

    # 处理第二个 JSON 文件
    for item in data2:
        image_path = item['image']
        print(image_path)
        hospital, patient_name = get_patient_hospital_info_3d(image_path)
        key = (hospital, patient_name)
        # if key not in merged_data:
        #     merged_data[key] = {'image_2d': [], 'image_3d': [], 'conversations_3d': item.get('conversations', [])}
        merged_data[key]['image_3d'].append(image_path)
        merged_data[key]['conversations_3d'] = item['conversations']

    # 将合并后的数据写入新的 JSON 文件
    merged_json = []
    for key, value in merged_data.items():
        merged_item = {
            'id': str(uuid.uuid4()),  # 为每个条目生成新的唯一 ID
            'hospital': key[0],
            'patient_name': key[1],
            'image_2d': value['image_2d'],
            'image_3d': value['image_3d'],
            'conversations_2d': value['conversations_2d'],  # 保留原来的 conversations
            'conversations_3d': value['conversations_3d']  # 保留原来的 conversations

        }
        merged_json.append(merged_item)

    # 保存合并后的结果
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(merged_json, out_f, ensure_ascii=False, indent=4)

    print(f"合并完成，结果已保存到 {output_file}")

# 示例调用
merge_json_files('/home/ly/LLMs/LongLLaVA-3D/data/train_data_2D_all.json', '/home/ly/LLMs/LongLLaVA-3D/data/train_data_3D_all.json', '/home/ly/LLMs/LongLLaVA-3D/data/merged_output.json')
