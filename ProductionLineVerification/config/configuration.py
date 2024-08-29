import cv2
import numpy as np
from roboflow import Roboflow

"""
PROGRAM OUTINE 
--> THREE CLASSES (INDEPENDENT OF EACH OTHER AT THIS POINT OF DEV):
1. BlueWasherDetect : ParamList : imaage_path
2. YellowWasherDetect : ParamList : image_path
3. BLACK AND WHITE WASHER (YOLO MODEL) : return type--> bool : ParamList : image_path
"""
#from memory_profiler import profile

#file = open("mem.log",'w')

# class BlueWasherDetect():
#     def __init__(self , image_path):
#         self.image_path = image_path

#     #@profile(stream = file)
#     # def detect_washer(self , lower=np.array([20, 80, 50]), upper=np.array([180, 255, 255])):
#     #     frame = cv2.imread(self.image_path)
#     #     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     #     mask = cv2.inRange(hsv, lower, upper)
#     #     # mask = cv2.GaussianBlur(mask, (5, 5), 0)
#     #     # mask = cv2.Canny(mask, 50, 150)
#     #     #cv2.imshow("Mask", mask)
#     #     # cv2.waitKey(0)
#     #     # cv2.destroyAllWindows()
#     #     circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=10, maxRadius=200)
    
#     #     if circles is not None:
#     #         circles = np.uint16(np.around(circles))
#     #         for i in circles[0, :]:
#     #             cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     #             cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
#     #         print("True" )
#     #     else:
#     #         print( "False"),None
#     def detect_washer(self, lower=np.array([94, 80, 2]), upper=np.array([130, 255, 255])):
#         frame = cv2.imread(self.image_path)
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, lower, upper)
#         circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=10, maxRadius=200)
    
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for i in circles[0, :]:
#                 cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
#                 cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
#             return True
#         else:
#             return False
#     # file2 = open("orientation.log" , 'w')
#     # @profile(stream = file2)
#     def check_orientation(self):
#         frame = cv2.imread(self.image_path)
#         # if frame is None:
#         #     return "Error: Could not read the image"

#         # color_ranges = {
#         #     'blue': (np.array([100, 150, 50]), np.array([140, 255, 255])),
#         #     'silver': (np.array([0, 0, 168]), np.array([180, 20, 255])),
#         # }

#         # roi_x, roi_y, roi_w, roi_h = 100, 100, 260, 240
#         # roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
#         # hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

#         # blue_mask = cv2.inRange(hsv, *color_ranges['blue'])
#         # silver_mask = cv2.inRange(hsv, *color_ranges['silver'])

#         # contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#         # if len(contours) > 0:
#         #     largest_contour = max(contours, key=cv2.contourArea)
#         #     contour_mask = np.zeros_like(blue_mask)
#         #     cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
#         #     silver_in_blue = cv2.bitwise_and(silver_mask, silver_mask, mask=contour_mask)
#         #     silver_count = cv2.countNonZero(silver_in_blue)
        
#         #     if silver_count > 500:
#         #         #print("The orientation of the blue washer is Downside")
#         #         return False
#         #     else:
#         #         #print("The orientation of the blue washer is Upside")
#         #         return True
#         # else:
#         #     return False 
#         #     #print("False")
#         #     #print("Detected yellow washer")
#         #     # file_name = f'yellowwasher_{len(os.listdir(yellowasher_dir))}.jpg'
#         #     # full_path = os.path.join(yellowasher_dir, file_name)
#         #     # save_yellow_washer_image(frame, roi_x, roi_y, roi_w, roi_h, full_path)
#         #     # upload_image_to_api("DETECTED YELLOW WASHER")

#         color_ranges = {
#                         'blue': (np.array([100, 150, 50]), np.array([140, 255, 255])),
#                         }

#         roi_x, roi_y, roi_w, roi_h = 100, 100, 260, 240

#         def detect_silver(frame):
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             lower_silver = 160
#             upper_silver = 255
#             mask = cv2.inRange(gray, lower_silver, upper_silver)
#             return mask

#         def check_hidden_cover_two(blue_mask, silver_mask):
#             blue_present = cv2.countNonZero(blue_mask) > 500
#             silver_present = cv2.countNonZero(silver_mask) > 500
#             return blue_present and silver_present

#         def analyze_frame(frame):
#             roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
#             hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
#             blue_mask = None
#             silver_mask = detect_silver(roi_frame)

#             for color_name, (lower, upper) in color_ranges.items():
#                 mask = cv2.inRange(hsv, lower, upper)
#                 if color_name == 'blue':
#                     blue_mask = mask

#             if blue_mask is not None:
#                 contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 if len(contours) > 0:
#                     largest_contour = max(contours, key=cv2.contourArea)
#                     contour_mask = np.zeros_like(blue_mask)
#                     cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
#                     inverted_contour_mask = cv2.bitwise_not(contour_mask)
#                     gap_in_contour = cv2.bitwise_and(blue_mask, blue_mask, mask=inverted_contour_mask)
#                     gap_count = cv2.countNonZero(gap_in_contour)
#                     if gap_count > 500:
#                         return False
#                     else:
#                         return True
class BlueWasherDetect:
    def __init__(self, image_path):
        self.image_path = image_path

    def detect_washer(self, lower=np.array([94, 80, 2]), upper=np.array([130, 255, 255])):
        frame = cv2.imread(self.image_path)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=315, maxRadius=395)
    
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            return True
        else:
            return False

    def check_orientation(self):
        frame = cv2.imread(self.image_path)
        
        color_ranges = {
            'blue': (np.array([100, 150, 50]), np.array([140, 255, 255])),
        }
        
        roi_x, roi_y, roi_w, roi_h = 100, 100, 260, 240
        roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        
        lower_silver = 160
        upper_silver = 255
        silver_mask = cv2.inRange(gray, lower_silver, upper_silver)
        
        blue_mask = None
        for color_name, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            if color_name == 'blue':
                blue_mask = mask

        def analyze_frame():
            if blue_mask is not None:
                contours, _ = cv2.findContours(blue_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    contour_mask = np.zeros_like(blue_mask)
                    cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                    inverted_contour_mask = cv2.bitwise_not(contour_mask)
                    gap_in_contour = cv2.bitwise_and(blue_mask, blue_mask, mask=inverted_contour_mask)
                    gap_count = cv2.countNonZero(gap_in_contour)
                    return gap_count <= 500
            return False
        
        def check_hidden_cover_two():
            blue_present = cv2.countNonZero(blue_mask) > 500 if blue_mask is not None else False
            silver_present = cv2.countNonZero(silver_mask) > 500
            return blue_present and silver_present
        
        frame_analysis_passed = analyze_frame()
        hidden_cover_passed = check_hidden_cover_two()
        
        return hidden_cover_passed and frame_analysis_passed

    def combined_result(self):
        washer_result = self.detect_washer()
        orientation_result = self.check_orientation()
        return washer_result and orientation_result



class YellowWasherDetect():
    def __init__(self , image_path):
        self.image_path = image_path
    def detect_washer(self, lower=np.array([20, 100, 100]), upper=np.array([30, 255, 255])):
        frame = cv2.imread(self.image_path)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=315, maxRadius=396)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            return True#, circles[0]
            #print('TRUE')
        else:
            return False#, None
            #print("FALSE")
        

from ultralytics import YOLO

class blackWhiteDetect():
    def __init__(self , image_path):
        self.image_path = image_path
    # file3 = open("black.log" , 'w')

    # @profile(stream = file3)
    # def BlackWhiteCheck(self):
    #     """Checks black and white using a pre-trained YOLO model."""
    #     model = YOLO('models/blackwhite.pt')
    #     img = cv2.imread(self.image_path)
    #     if img is None:
    #         print(f"Failed to load image from {self.image_path}. Please check the file path.")
    #         return None
    #     results = model(img)
    #     if results and len(results) > 0:
    #         result = results[0]
    #         if len(result.boxes) > 0:
    #             box = result.boxes[0]
    #             class_id = int(box.cls[0])
    #             confidence = float(box.conf[0])
    #             orientation_labels = {0: 'CORRECT', 1: 'CORRECT-NOTCORRECT',2:'NOTCORRECT'} 
    #             orientation_label = orientation_labels.get(class_id, 'Unknown')
            
    #             print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
    #             return orientation_label
    #         else:
    #             return True
    #             #return None
    #     else:
    #         return False
            #return None
    
    def BlackWhiteCheck(self):
        rf = Roboflow(api_key="HybCHsLMVoI0IBhwRGJk")
        project = rf.workspace().project("blackwhite-cw5fv")
        model = project.version(1).model

        # infer on a local image
        result = model.predict("resources/IMG_0811.jpg", confidence=1, overlap=50).json()
        print(result['predictions'][0]['class'])
        if result == 'correct':
            return True
        else:
            return False



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
