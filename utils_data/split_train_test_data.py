import json
import random
import os

def get_patient_hospital_info(image_path):
    # 提取图片路径中的倒数第三层级和倒数第二层级，分别为医院和病人名字
    path_parts = image_path[0][0].split(os.sep)
    hospital = path_parts[-4]  # 倒数第三层级
    patient_name = path_parts[-3]  # 倒数第二层级
    return hospital, patient_name

def split_dataset(input_json, output_train, output_test):
    # 读取融合后的 JSON 文件
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 按医院分组（合并 zhongsaneryuan 和 zhongsaneryuan2）
    hospital_groups = {}
    for item in data:
        # image_path = item['image_2d']
        # hospital,_ = get_patient_hospital_info(image_path)
        hospital = item['hospital']
        # print(hospital)
        if hospital in ['zhongsaneryuan', 'zhongsaneryuan2']:
            hospital = 'zhongsaneryuan_combined'
        
        if hospital not in hospital_groups:
            hospital_groups[hospital] = []
        
        hospital_groups[hospital].append(item)
    # 筛选测试集（每个医院10例，包含T2A.npy）
    test_set = []
    for hospital, patients in hospital_groups.items():
        # 筛选包含 T2A.npy 的病人
        # hospital = 
        # print(patients[0]['image_3d'])
        patients_with_t2a = [p for p in patients if any('T2A.npy' in img for img in p['image_3d'][0])]
        random.shuffle(patients_with_t2a)
        # print(patients)
        # 选取最多10例
        selected_test_patients = patients_with_t2a[:10]
        test_set.extend(selected_test_patients)
        
        # 从原分组中移除测试集病人
        hospital_groups[hospital] = [p for p in patients if p not in selected_test_patients]
    
    # 剩下的作为训练集
    train_set = [p for patients in hospital_groups.values() for p in patients]
    
    # 保存训练集
    with open(output_train, 'w', encoding='utf-8') as f:
        json.dump(train_set, f, ensure_ascii=False, indent=4)
    
    # 保存测试集
    with open(output_test, 'w', encoding='utf-8') as f:
        json.dump(test_set, f, ensure_ascii=False, indent=4)
    
    print(f"训练集保存到: {output_train}")
    print(f"测试集保存到: {output_test}")


# 示例调用
split_dataset(
    input_json='/home/ly/LLMs/LongLLaVA-3D/data/merged_output_all.json',
    output_train='/home/ly/LLMs/LongLLaVA-3D/data/train_split.json',
    output_test='/home/ly/LLMs/LongLLaVA-3D/data/test_split.json'
)
