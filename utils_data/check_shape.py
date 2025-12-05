import numpy as np

def check_npy_shape(file_path):
    """
    读取 .npy 文件并打印其形状。
    
    :param file_path: str, .npy 文件路径
    """
    try:
        data = np.load(file_path)  # 加载 .npy 文件
        print(f"文件: {file_path}")
        print(f"数据类型: {type(data)}")
        print(f"数据形状: {data.shape}")
    except Exception as e:
        print(f"无法读取文件 {file_path}，错误: {e}")

if __name__ == "__main__":
    # 修改为你的 .npy 文件路径
    npy_file_path = "/home/zwding/hospital_data_3D/zongliuyiyuan/10463269/DWI.npy"
    check_npy_shape(npy_file_path)
