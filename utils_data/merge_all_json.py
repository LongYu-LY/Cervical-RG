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


def merge_json_files(json_files, output_file):
    # 创建一个字典，以医院和病人名字为键，存储对应的图片路径
    merged_data = {}

    # 循环处理所有 JSON 文件
    for i,json_file in enumerate(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # print(data)
        # 处理每个 JSON 文件的内容
        for item in data:
            image_path = item['image']
            if i == 0:
                hospital, patient_name = get_patient_hospital_info(image_path)
                key = (hospital, patient_name)
                if key not in merged_data:
                    merged_data[key] = {'image_2d': [], 'image_3d': [], 'image_2d_GPT4': [], 'image_3D_T2A': [], 'image_3D_T2A_T1CA_ADC': [], 
                                        'conversations_2d': [], 'conversations_3d': [], 'conversations_2d_GPT4': [], 'conversations_3D_T2A': [], 'conversations_3D_T2A_T1CA_ADC': []}
                merged_data[key]['image_2d'].append(image_path)
                merged_data[key]['conversations_2d'] = item['conversations']
            elif i == 1:
                # print(image_path)
                hospital, patient_name = get_patient_hospital_info_3d(image_path)
                key = (hospital, patient_name)
                merged_data[key]['image_3d'].append(image_path)
                merged_data[key]['conversations_3d'] = item['conversations']
            elif i == 2:
                hospital, patient_name = get_patient_hospital_info(image_path)
                key = (hospital, patient_name)
                merged_data[key]['image_2d_GPT4'].append(image_path)
                merged_data[key]['conversations_2d_GPT4'] = item['conversations']
            elif i == 3:
                hospital, patient_name = get_patient_hospital_info_3d(image_path)
                key = (hospital, patient_name)
                merged_data[key]['image_3D_T2A'].append(image_path)
                merged_data[key]['conversations_3D_T2A'] = item['conversations']
            elif i == 4:
                hospital, patient_name = get_patient_hospital_info_3d(image_path)
                key = (hospital, patient_name)
                merged_data[key]['image_3D_T2A_T1CA_ADC'].append(image_path)
                merged_data[key]['conversations_3D_T2A_T1CA_ADC'] = item['conversations']

    # 将合并后的数据写入新的 JSON 文件
    merged_json = []
    for key, value in merged_data.items():
        merged_item = {
            'id': str(uuid.uuid4()),  # 为每个条目生成新的唯一 ID
            'hospital': key[0],
            'patient_name': key[1],
            'image_2d': value['image_2d'],
            'image_3d': value['image_3d'],
            'image_2d_GPT4': value['image_2d_GPT4'],
            'image_3D_T2A': value['image_3D_T2A'],
            'image_3D_T2A_T1CA_ADC': value['image_3D_T2A_T1CA_ADC'],
            'conversations_2d': value['conversations_2d'],  # 保留原来的 conversations
            'conversations_3d': value['conversations_3d'],
            'conversations_2d_GPT4': value['conversations_2d_GPT4'],
            'conversations_3D_T2A': value['conversations_3D_T2A'],
            'conversations_3D_T2A_T1CA_ADC': value['conversations_3D_T2A_T1CA_ADC']
        }
        merged_json.append(merged_item)

    # 保存合并后的结果
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(merged_json, out_f, ensure_ascii=False, indent=4)

    print(f"合并完成，结果已保存到 {output_file}")

# 示例调用：合并 5 个 JSON 文件
merge_json_files([
    '/home/ly/LLMs/LongLLaVA-3D/data/train_data_2D_all.json',
    '/home/ly/LLMs/LongLLaVA-3D/data/train_data_3D_all.json',
    '/home/ly/LLMs/LongLLaVA-3D/data/train_data_2D_2_all.json',
    '/home/ly/LLMs/LongLLaVA-3D/data/train_data_T2A_all.json',
    '/home/ly/LLMs/LongLLaVA-3D/data/train_data_T2A_T1CA_ADC_all.json'
], '/home/ly/LLMs/LongLLaVA-3D/data/merged_output_all.json')
