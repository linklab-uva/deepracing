import pyf1_datalogger, numpy as np, cv2
dl = pyf1_datalogger.ScreenVideoCapture()

dl.open("F1 2017",100,100,500,500)

im = dl.read()
print(np.mean(im))
cv2.namedWindow("F1 2017",0)
cv2.imshow("F1 2017",im)
cv2.waitKey(0)

