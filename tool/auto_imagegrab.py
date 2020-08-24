
import time
import os
import win32api
import win32con
from PIL import ImageGrab

directory = 'captured_picture'
if not os.path.exists(directory):
    os.mkdir(directory)
    print('3秒后开始滚动截图…')
time.sleep(3)
for i in range(100):
    ImageGrab.grab((300,100,1700,900)).save(rf"{directory}\ttt{i}.jpg")
    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, -250)
    time.sleep(0.1)

