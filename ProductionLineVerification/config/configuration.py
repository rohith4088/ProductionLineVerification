import cv2
import numpy as np
import os
import requests

"""
PROGRAM OUTINE 
--> THREE CLASSES (INDEPENDENT OF EACH OTHER AT THIS POINT OF DEV):
1. BlueWasherDetect : ParamList : imaage_path
2. YellowWasherDetect : ParamList : image_path
3. BLACK AND WHITE WASHER (YOLO MODEL) : return type--> bool : ParamList : image_path
"""
from memory_profiler import profile

#file = open("mem.log",'w')

class BlueWasherDetect():
    def __init__(self , image_path):
        self.image_path = image_path

    #@profile(stream = file)
    # def detect_washer(self , lower=np.array([20, 80, 50]), upper=np.array([180, 255, 255])):
    #     frame = cv2.imread(self.image_path)
    #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(hsv, lower, upper)
    #     # mask = cv2.GaussianBlur(mask, (5, 5), 0)
    #     # mask = cv2.Canny(mask, 50, 150)
    #     #cv2.imshow("Mask", mask)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=10, maxRadius=200)
    
    #     if circles is not None:
    #         circles = np.uint16(np.around(circles))
    #         for i in circles[0, :]:
    #             cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #             cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    #         print("True" )
    #     else:
    #         print( "False"),None
    def detect_washer(self, lower=np.array([20, 80, 50]), upper=np.array([180, 255, 255])):
        frame = cv2.imread(self.image_path)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=10, maxRadius=200)
    
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            print("True")
        else:
            print("False")
    # file2 = open("orientation.log" , 'w')
    # @profile(stream = file2)
    def check_orientation(self):

        frame = cv2.imread(self.image_path)
        if frame is None:
            return "Error: Could not read the image"

        color_ranges = {
            'blue': (np.array([100, 150, 50]), np.array([140, 255, 255])),
            'silver': (np.array([0, 0, 168]), np.array([180, 20, 255])),
        }

        roi_x, roi_y, roi_w, roi_h = 100, 100, 260, 240
        roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, *color_ranges['blue'])
        silver_mask = cv2.inRange(hsv, *color_ranges['silver'])

        contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_mask = np.zeros_like(blue_mask)
            cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            silver_in_blue = cv2.bitwise_and(silver_mask, silver_mask, mask=contour_mask)
            silver_count = cv2.countNonZero(silver_in_blue)
        
            if silver_count > 500:
                print("The orientation of the blue washer is Downside")
                #return False
            else:
                print("The orientation of the blue washer is Upside")
                #return True
        else:
            print("Detected yellow washer")
            # file_name = f'yellowwasher_{len(os.listdir(yellowasher_dir))}.jpg'
            # full_path = os.path.join(yellowasher_dir, file_name)
            # save_yellow_washer_image(frame, roi_x, roi_y, roi_w, roi_h, full_path)
            # upload_image_to_api("DETECTED YELLOW WASHER")

class YellowWasherDetect():
    def __init__(self , image_path):
        self.image_path = image_path
    def detect_washer(self, lower=np.array([15, 150, 150]), upper=np.array([35, 255, 255])):
        frame = cv2.imread(self.image_path)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=10, maxRadius=200)
    
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            return True, circles[0]
        else:
            return False, None
        

from ultralytics import YOLO

class blackWhiteDetect():
    def __init__(self , image_path):
        self.image_path = image_path
    # file3 = open("black.log" , 'w')

    # @profile(stream = file3)
    def BlackWhiteCheck(self):
        """Checks black and white using a pre-trained YOLO model."""
        model = YOLO('models/blackwhite.pt')
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"Failed to load image from {self.image_path}. Please check the file path.")
            return None
        results = model(img)
        if results and len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                box = result.boxes[0]
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                orientation_labels = {0: 'CORRECT', 1: 'CORRECT-NOTCORRECT',2:'NOTCORRECT'} 
                orientation_label = orientation_labels.get(class_id, 'Unknown')
            
                print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
                return orientation_label
            else:
                print("No washer detected.")
                return None
        else:
            print("No detections made.")
            return None


# class PistonDetect():
#     def __init__(self , image_path):
#         self.image_path = image_path
#     def PistonCheck(self):
#         """Checks black and white using a pre-trained YOLO model."""
#         model = YOLO('models/piston.pt')
#         img = cv2.imread(self.image_path)
#         if img is None:
#             print(f"Failed to load image from {self.image_path}. Please check the file path.")
#             return None
#         results = model(img)
#         if results and len(results) > 0:
#             result = results[0]
#             if len(result.boxes) > 0:
#                 box = result.boxes[0]
#                 class_id = int(box.cls[0])
#                 #print(class_id)
#                 confidence = float(box.conf[0])
#                 orientation_labels = {1:"piston"} 
#                 orientation_label = orientation_labels.get(class_id, 'Unknown')
            
#                 print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
#                 return orientation_label
#             else:
#                 print("No piston detected.")
#                 return None
#         else:
#             print("No detections made.")
#             return None


# obj = BlueWasherDetect("/Users/rohithr/Desktop/wipro_clean/bluewasher/blue_washer.jpg")
# obj.detect_washer()
# print(obj.check_orientation())


# bwo = blackWhiteDetect("/Users/rohithr/Desktop/wipro_clean/resources/bw.jpeg")
# bwo.BlackWhiteCheck()
