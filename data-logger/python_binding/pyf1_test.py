import pyf1_datalogger, numpy as np, cv2
dl = pyf1_datalogger.ScreenVideoCapture()
app="F1 2017"
dl.open(app,0,200,1700,300)
cv2.namedWindow(app,cv2.WINDOW_AUTOSIZE)
while True:
    im = dl.read()
    cv2.imshow(app,im)
    cv2.waitKey(5)

