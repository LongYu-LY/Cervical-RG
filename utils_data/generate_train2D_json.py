import json
import csv
import random
import os
import re

def generate_id(index):
    # 生成一个12位的ID，不足部分补零
    return str(index).zfill(12)

data = []
test_data = []
questions = [
        "We will provide you with various medical images of some patients. We kindly ask you to provide a diagnosis based on these images.",
        "You will receive some multimodal medical images of patients. We hope you can provide a diagnosis based on these images.",
        "We will present you with different types of medical images of some patients. We kindly ask you to make a diagnosis based on these images.",
        "You will receive various multimodal medical images of patients. We request you to provide a diagnosis based on these images.",
        "We plan to provide you with some multimodal medical images of patients. We kindly ask you to give a diagnostic opinion on them.",
        "We will provide you with a set of multimodal medical images of several patients. We hope you can make a diagnosis based on these images.",
        "You will receive some multimodal medical images of patients. We look forward to your diagnostic assessment.",
        "We will share with you different types of medical images of some patients. We look forward to your diagnostic opinion.",
        "We will provide you with various multimodal medical images of some patients. We kindly ask you to provide a diagnosis based on them.",
        "You will receive some multimodal medical images of patients. We request your diagnostic evaluation."
]

# 正则表达式提取文件名中尾部的数字部分
def extract_last_number(filename):
    match = re.search(r'(\d+)(?=\D*$)', filename)
    return int(match.group()) if match else float('inf')

# 指定包含多个医院数据的主文件夹路径
main_folder_path = "/home/ly/LLMs/LongLLaVA/new_LLM_data/hospital_data_2D/"


# 遍历主文件夹中的每个医院目录
# 为每个医院单独存储数据
hospital_data = []

# 遍历每个医院目录
i = 1
for hospital_folder in os.listdir(main_folder_path):
    hospital_path = os.path.join(main_folder_path, hospital_folder)

    # 查找医院目录下的所有 CSV 文件
    csv_files = [f for f in os.listdir(hospital_path) if f.endswith('.csv')]
    if not csv_files:
        print(f"{hospital_folder} 中未找到 CSV 文件，跳过该医院。")
        continue

    # 为每个医院创建临时数据存储
    hospital_entries = []

    for csv_file in csv_files:
        csv_file_path = os.path.join(hospital_path, csv_file)
        
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # 跳过表头行
            
            for row in csv_reader:
                image_folder_path = os.path.join(hospital_path, row[1])  
                
                # 对每个模态文件夹中的图片按尾部数字排序，并取中间部分10张
                image_paths = []
                modality_images = []
                for root, _, files in os.walk(image_folder_path):
                    modality = os.path.basename(root)  # 模态名使用父文件夹名
                    current_modality_images = [os.path.join(root, f) for f in files if f.lower().endswith('.png')]
                    current_modality_images.sort(key=lambda x: extract_last_number(os.path.basename(x)))
                    
                    # total_images = len(current_modality_images)
                    # start_index = max((total_images - 10) // 2, 0)
                    # end_index = min(start_index + 10, total_images)
                    total_images = len(current_modality_images)
                    start_index = max((total_images - 2) // 2, 0)  # 获取最中间的两个元素前面的元素的索引
                    end_index = start_index + 2  # 因为我们需要2个元素
                    selected_images = current_modality_images[start_index:end_index]
                    image_paths.extend(selected_images)
                    modality_images.extend([(modality, img) for img in selected_images])

                random_sentence = random.choice(questions)
                
                # 为每张图片标注模态并在对话中表明
                modality_description = '\n'.join([f"{modality}: <image>" for modality, img in modality_images])
                if len(image_paths) == 0:
                    print('剔除',image_folder_path)
                    continue
                entry = {
                    "id": generate_id(i),
                    "image": image_paths,  
                    "conversations": [
                        {
                            "from": "human",
                            "value": random_sentence + '\n' + 
                                                "These images are: " +  ''.join([f"<image>" for _ in range(len(image_paths))]) + '\n' +
                                                ("Based on the provided information, there are no metastatic lymph nodes." if row[2] == '0' 
                                                else "Based on the provided information, there are metastatic lymph nodes.")
                        },
                        {
                            "from": "gpt",
                            "value": "Our medical diagnosis is: " + row[3]
                        }
                    ],
                }
                hospital_entries.append(entry)
                i += 1

    # 随机分出5条数据作为测试集，其余为训练集
    random.shuffle(hospital_entries)
    test_set = hospital_entries[:5]
    train_set = hospital_entries[5:]

    # 将结果加入总数据
    data.extend(train_set)
    data.extend(test_set)
    test_data.extend(test_set)
    # 如果需要单独保存测试集数据


# 输出所有训练数据
output_json_path = "/home/ly/LLMs/LongLLaVA-3D/data/train_data_2D_10_all.json"
with open(output_json_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=2, ensure_ascii=False)

print(f"所有训练数据已成功写入 {output_json_path}。")
print(f"共生成 {len(data)} 条训练数据。")

test_output_path = '/home/ly/LLMs/LongLLaVA-3D/data/test_data_2D.json'
with open(test_output_path, 'w', encoding='utf-8') as test_file:
    json.dump(test_data, test_file, indent=2, ensure_ascii=False)
print(f"测试集已写入 {test_output_path}")