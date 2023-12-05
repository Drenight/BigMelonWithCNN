# install npm first
# then install serve
# cd to the game of file and run command "serve" to open the game locally

import json
import time
import random
import selenium
from selenium import webdriver
import time
import os
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options

# emulate chrome as mobile, i.e., make the website 
# just like the mobile phone to run the game. 

mobile_emulation = {
   "deviceMetrics": { "width": 360, "height": 640, "pixelRatio": 3.0 },
   "userAgent": "Mozilla/5.0 (Linux; Android 4.2.1; en-us; Nexus 5 Build/JOP40D) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19",
   "clientHints": {"platform": "Android", "mobile": True} 
}

chrome_options = Options()

chrome_options.add_experimental_option("mobileEmulation", mobile_emulation)

driver = webdriver.Chrome(options = chrome_options)


# driver.get(your html)
driver.get('http://localhost:3000')
time.sleep(4)


# driver.get_screenshot_as_file(path/to/figure)
# take screenshot
# !!! ONLY PNG format works !!!

step_count = 0
# step_count_str = '{:03d}'.format(step_count)

# 获取当前时间戳
timestamp = str(int(time.time()))
directory_path = os.getcwd()+'/../original_data/' + timestamp
os.makedirs(directory_path, exist_ok=False)

coor_json = {}

while step_count<300:
   # fn_name = os.path.join(os.getcwd(), '..', 'original_data', 'play_' + str(timestamp), 'tryres' + str(step_count) + '.png')
   # fn_name = directory_path+'/'+str(step_count)+'.png'
   # driver.get_screenshot_as_file(fn_name)

   # move the mouse to the position (15,80)
   # 往右15往下80

   por1 = random.randint(0,100) 
   coor1 = 15 + 0.01*por1*(340-15)
   por2 = random.randint(0,100)
   coor2 = 15 + 0.01*por2*(340-15)

   coor_json[step_count+1] = por1
   coor_json[step_count+2] = por2

   file_path = directory_path+"/coor.json"

   # 将字典写入 JSON 文件
   with open(file_path, 'w') as json_file:
      json.dump(coor_json, json_file)

   step_count += 1
   ActionChains(driver).move_by_offset(coor1 ,80).click().perform()
   # since here is the relative position, if not move back, 
   # it will continuously apply the movement.
   ActionChains(driver).move_by_offset(-coor1,-80).perform()

   # sleep sometime to wait for fruits falling down
   time.sleep(2)
   # driver.get_screenshot_as_file(os.getcwd()+'/tryres.png')
   fn_name = directory_path+'/'+str(step_count)+'.png'
   driver.get_screenshot_as_file(fn_name)


   step_count += 1
   # time.sleep(2)
   ActionChains(driver).move_by_offset(coor2,80).click().perform()
   ActionChains(driver).move_by_offset(-coor2,-80).perform()
   time.sleep(2)
   # driver.get_screenshot_as_file(os.getcwd()+'/tryres2.png')
   # step_count += 1
   fn_name = directory_path+'/'+str(step_count)+'.png'
   driver.get_screenshot_as_file(fn_name)

# or driver.quit() to close all the opening windows.
driver.close()
