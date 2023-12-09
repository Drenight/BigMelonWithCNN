import os 
import json

directory_path = "./original_data"
file_names = os.listdir(directory_path)

image_files = []
for file in file_names:
    folder_path = os.path.join(directory_path, file)
    if os.path.isdir(folder_path):
        # Enter each running
        score = os.path.join(folder_path, "scores.json")
        with open(score, 'r') as json_file:
            data = json.load(json_file)
            print(data)
            exit()

        files_in_folder = os.listdir(folder_path)
        for f in files_in_folder:
            file_path = os.path.join(folder_path, f)
            if os.path.isfile(file_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(file_path)
            else:
                print(f)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    print(data)
        # print(len(image_files))
    else:
        # Alert
        pass
print(str(len(image_files)) + " images loaded!")