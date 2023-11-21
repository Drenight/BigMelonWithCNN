# install npm first
# then install serve
# cd to the game of file and run command "serve" to open the game locally

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

while True:
   step_count += 1
   driver.get_screenshot_as_file(os.getcwd()+'/tryres'+str(step_count)+'.png')

   # move the mouse to the position (15,80)
   # 往右15往下80
   ActionChains(driver).move_by_offset(15,80).click().perform()

   # since here is the relative position, if not move back, 
   # it will continuously apply the movement.
   ActionChains(driver).move_by_offset(-15,-80).perform()

   # sleep sometime to wait for fruits falling down
   time.sleep(2)


   # driver.get_screenshot_as_file(os.getcwd()+'/tryres.png')
   step_count += 1
   driver.get_screenshot_as_file(os.getcwd()+'/tryres'+str(step_count)+'.png')


   time.sleep(2)
   ActionChains(driver).move_by_offset(340,80).click().perform()
   ActionChains(driver).move_by_offset(-340,-80).perform()
   time.sleep(2)
   # driver.get_screenshot_as_file(os.getcwd()+'/tryres2.png')
   step_count += 1
   driver.get_screenshot_as_file(os.getcwd()+'/tryres'+str(step_count)+'.png')



# or driver.quit() to close all the opening windows.
driver.close()
