import json
import os

# 输入和输出文件路径
input_json_path = '/home/ly/LLMs/LongLLaVA-3D/data/train_split.json'
output_dir = '/home/ly/LLMs/LongLLaVA-3D/json_data/train_data'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取原始JSON文件
with open(input_json_path, 'r', encoding='utf-8') as f:
    datas = json.load(f)

# 用来存储所有处理后的数据
processed_data = {
    "image_2d_GPT4_conversations_2d_GPT4": [],
    "image_2d_conversations_2d": [],
    "image_3D_T2A_T1CA_ADC_conversations_3D_T2A_T1CA_ADC": [],
    "image_3D_T2A_conversations_3D_T2A": [],
    "image_3D_conversations_3D": []
}

# 遍历每个数据块并处理
for data in datas:
    # 提取公共信息：id、hospital 和 patient_name
    common_info = {
        "id": data.get("id"),
        "hospital": data.get("hospital"),
        "patient_name": data.get("patient_name")
    }

    # 1. image_2d_GPT4 + conversations_2d_GPT4
    data_2d_gpt4 = {
        **common_info,
        "image": data.get("image_2d_GPT4", [])[0],
        "conversations": data.get("conversations_2d_GPT4", [])
    }
    processed_data["image_2d_GPT4_conversations_2d_GPT4"].append(data_2d_gpt4)

    # 2. image_2d + conversations_2d
    data_2d = {
        **common_info,
        "image": data.get("image_2d", [])[0],
        "conversations": data.get("conversations_2d", [])
    }
    processed_data["image_2d_conversations_2d"].append(data_2d)

    # 3. image_3D_T2A_T1CA_ADC + conversations_3D_T2A_T1CA_ADC
    data_3d_t2a_t1ca_adc = {
        **common_info,
        "image": data.get("image_3D_T2A_T1CA_ADC", [])[0] if data.get("image_3D_T2A_T1CA_ADC", []) is list else data.get("image_3D_T2A_T1CA_ADC", []),
        "conversations": data.get("conversations_3D_T2A_T1CA_ADC", [])
    }
    processed_data["image_3D_T2A_T1CA_ADC_conversations_3D_T2A_T1CA_ADC"].append(data_3d_t2a_t1ca_adc)

    # 4. image_3D_T2A + conversations_3D_T2A
    data_3d_t2a = {
        **common_info,
        "image": data.get("image_3D_T2A", [])[0] if data.get("image_3D_T2A", []) is list else data.get("image_3D_T2A", []),
        "conversations": data.get("conversations_3D_T2A", [])
    }
    processed_data["image_3D_T2A_conversations_3D_T2A"].append(data_3d_t2a)

    # 5. image_3D + conversations_3D
    data_3d = {
        **common_info,
        "image": data.get("image_3d", [])[0],
        "conversations": data.get("conversations_3d", [])
    }
    processed_data["image_3D_conversations_3D"].append(data_3d)

# 在所有数据处理完后一次性写入文件
for category, data in processed_data.items():
    with open(os.path.join(output_dir, f'{category}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

print("JSON数据已成功按五类拆分并处理完毕，所有数据已写入文件。")
