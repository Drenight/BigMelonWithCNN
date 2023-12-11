import os
import json
import shutil

def main():
    # 设置原始数据文件夹和目标文件夹路径
    root_original_data_folder = './original_data'
    positive_data_folder = './positive_data'

    # 确保positive_data文件夹存在
    os.makedirs(positive_data_folder, exist_ok=True)

    # 遍历original_data文件夹中的所有子文件夹
    for subfolder in os.listdir(root_original_data_folder):
        original_data_folder = os.path.join(root_original_data_folder, subfolder)
        if os.path.isdir(original_data_folder):
            timestamps_file_path = os.path.join(original_data_folder, 'timestamps.json')
            coor_file_path = os.path.join(original_data_folder, 'coor.json')

            if not os.path.exists(timestamps_file_path) or not os.path.exists(coor_file_path):
                print(f"Missing timestamps.json or coor.json in folder {subfolder}. Skipping.")
                continue

            # 读取时间戳文件和坐标文件
            with open(timestamps_file_path, 'r') as file:
                timestamps = json.load(file)
            with open(coor_file_path, 'r') as file:
                coor_dict = json.load(file)

            # 获取所有图片文件
            image_files = [f for f in os.listdir(original_data_folder) if f.endswith('.png')]
            # 将图片文件按照时间戳排序
            image_files.sort(key=lambda filename: int(filename.split('.')[0]))

            # 对于每个时间戳，找到对应的图片文件
            for timestamp in timestamps:
                corresponding_image = None
                for image_file in image_files:
                    image_timestamp = int(image_file.split('.')[0])
                    if image_timestamp <= timestamp:
                        corresponding_image = image_file
                    else:
                        break

                if corresponding_image:
                    # 获取coor字典中对应的坐标值
                    coor_value = coor_dict.get(corresponding_image.split('.')[0])
                    if coor_value is not None:
                        # 重命名图片文件，添加坐标值后缀
                        new_image_name = f"{corresponding_image.split('.')[0]}_{coor_value}.png"
                        # 构建完整的图片路径和目标路径
                        image_path = os.path.join(original_data_folder, corresponding_image)
                        target_path = os.path.join(positive_data_folder, new_image_name)
                        # 复制并重命名图片到positive_data文件夹
                        shutil.copy2(image_path, target_path)
                        print(f"Copied and renamed '{corresponding_image}' to '{new_image_name}' in positive_data folder.")
                    else:
                        print(f"No coordinate value found for image {corresponding_image} in coor.json.")
                else:
                    print(f"No image found for timestamp {timestamp} in folder {subfolder}.")





