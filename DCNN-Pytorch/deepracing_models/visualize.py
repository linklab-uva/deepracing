import numpy as np 
import cv2
import pyautogui
import time
dts = []
for i in range(100):
    tick = time.time()
    image = pyautogui.screenshot()
    imnp = np.array(image)
    tock = time.time()
    dts.append((tock-tick))
dtsnp = np.array(dts)
meandt = np.mean(dtsnp)
print("mean dt: %f" %(np.mean(dtsnp)))
print("frequency: %f" %(1.0/meandt))