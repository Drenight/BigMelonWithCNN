#from data_collector import try

import os
import shutil
from cnn import cnn
from data_collector import random_player
import filter 

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    else:
        print(f"文件夹不存在：{folder_path}")

clear_folder('./original_data')
clear_folder('./positive_data')

## Generate newdata until batch filled
#TODO
random_player.main()
filter.main()

## Train the model
cnn.main()
## End