import json

# 输入和输出文件路径
input_file = '/home/ly/LLMs/LongLLaVA-3D/json_data/train_data/train.json'  # 输入JSON文件路径
output_file = '/home/ly/LLMs/LongLLaVA-3D/json_data/train_data/train_3d.json'  # 输出JSON文件路径
new_base_path = '/home/ly/LLMs/M3D/Data/hospital_data_3D'  # 新的基础路径

# 读取JSON数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 保存JSON数据
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 修改路径函数
def update_image_paths(data):
    # print(data[0]['image'])
    for item in data:
        # if 'image' in item and isinstance(item['image'], list):
            for i, img_path in enumerate(item['image']):
                # if isinstance(img_list, list):
                #     for j, img_path in enumerate(img_list):
                print(img_path)
                if img_path.startswith('/home/ubuntu/zhw/hospital_data_3D/'):
                    updated_path = img_path.replace('/home/ubuntu/zhw/hospital_data_3D/', new_base_path + '/')
                    item['image'][i] = updated_path
    return data

# 主程序
if __name__ == '__main__':
    input_data = load_json(input_file)
    updated_data = update_image_paths(input_data)
    save_json(updated_data, output_file)
    print(f'路径已更新，结果已保存到 {output_file}')
