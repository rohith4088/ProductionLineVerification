import cv2
import time
from schedule import repeat , run_pending , run_all , every

cap = cv2.VideoCapture(0)

roi_x , roi_y , roi_w , roi_h = 100,100,300,300
#@repeat(every(3).seconds())
def capture():
    while True:
        ret , frame = cap.read()
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        cv2.imshow("frame" , frame)


capture()