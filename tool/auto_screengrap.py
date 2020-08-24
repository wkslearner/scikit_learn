'''
自动截屏程序
chromedriver 下载
https://sites.google.com/a/chromium.org/chromedriver/downloads
chromedriver 安装
https://blog.csdn.net/speverriver/article/details/78689722
教程
https://www.cnblogs.com/superhin/p/11482188.html
'''

import time
from time import sleep
from PIL import Image
import numpy as np
from selenium import webdriver

# # 网址截图抓取
# driver = webdriver.Chrome()
# driver.fullscreen_window()
# driver.implicitly_wait(3)
# driver.get("https://www.google.com.hk/search?newwindow=1&safe=strict&biw=1184&bih=551&ei=9gkVX-PnGMH6-QaE25jIDQ&q=python&oq=python&gs_lcp=CgZwc3ktYWIQAzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzIECAAQQzICCABQvPGHB1iMiIgHYK-RiAdoAnAAeAGAAbYFiAGZFJIBBzItMy41LTOYAQKgAQGqAQdnd3Mtd2l6sAEA&sclient=psy-ab&ved=0ahUKEwjjupC67NrqAhVBfd4KHYQtBtkQ4dUDCAw&uact=5")
# time.sleep(1)
#
# # 保存抓取的图片
# driver.get_screenshot_as_file("google.png")
# driver.quit()

'''页面滚动截屏'''
driver = webdriver.Chrome()
driver.fullscreen_window()  # 全屏窗口
driver.get('https://www.google.com.hk/search?safe=strict&source=hp&ei=nUcVX7HvFo_Dz7sP_vGv0A4&q=python&btnK=Google+%E6%90%9C%E7%B4%A2')
window_height = driver.get_window_size()['height']  # 窗口高度

page_height = driver.execute_script('return document.documentElement.scrollHeight')  # 页面高度
driver.save_screenshot('qq.png')

if page_height > window_height:
    n = page_height // window_height  # 需要滚动的次数
    base_mat = np.atleast_2d(Image.open('qq.png'))  # 打开截图并转为二维矩阵

    for i in range(n):
        driver.execute_script(f'document.documentElement.scrollTop={window_height*(i+1)};')
        sleep(.5)
        driver.save_screenshot(f'qq_{i}.png')  # 保存截图
        mat = np.atleast_2d(Image.open(f'qq_{i}.png'))  # 打开截图并转为二维矩阵
        base_mat = np.append(base_mat, mat, axis=0)  # 拼接图片的二维矩阵
    Image.fromarray(base_mat).save('hao123.png') # 图片拼接结果

driver.quit()






