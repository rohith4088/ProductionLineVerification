# import cv2 
# from ultralytics import YOLO
# class Detect():
#     def __init__(self , image_path):
#         self.image_path = image_path

#     def ComponentDetction(self):
#         model = YOLO('models/component.pt')
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
#                 confidence = float(box.conf[0])
#                 orientation_labels = {0: 'HIDDEN_COVER_TWO'} 
#                 orientation_label = orientation_labels.get(class_id, 'Unknown')
#                 #print(orientation_labels.values())
            
#                 #print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
#                 return [orientation_label,True]
#             else:
#                 return [orientation_label,False]
#                 #print("this is first else part")
#                 #print("this is ",orientation_labels.values())
#                 #return True
#         #         #return None
#         # else:
#         #     return ["HIDDEN_COVER_TWO",False]
#             #print("this is else part")
#             #print(orientation_label.vales())
#             # print("false")
#             # return False
# # det = Detect("resources/octogan.jpeg")
# # print(det.ComponentDetction())


# import cv2
# class ComponentDetect():
#     def __init__(self , image_path):
#         self.image_path = image_path

#     def DetectComponent(self):
#         img = cv2.imread(self.image_path)
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)
#         edges = cv2.Canny(blur, 50, 150)
#         contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True,approxCurve=cv2.arcLength(cnt , True))
#             if len(approx) == 8:
#                 cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
#                 cv2.imshow("Octagon Detection", img)
#         #         return ["HIDDEN_COVER_TWO", True]

#         # else:
#         #     return [None , False]

#         cv2.imshow("Octagon Detection", img)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
# cd = ComponentDetect("resources/octogan.jpeg")
# print(cd.DetectComponent())



import cv2
import numpy as np
class ComponentDetect():
    def __init__(self , image_path):
        self.image_path = image_path

    def DetectComponent(self):
        pixel_to_mm = 0.5
        min_diameter_mm = 10
        max_diameter_mm = 35
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        detected_circles = cv2.HoughCircles(gray_blurred,cv2.HOUGH_GRADIENT, 1, 20, param1=50,param2=30, minRadius=1, maxRadius=40)
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            count = 0
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
                diameter_in_mm = 2 * r * pixel_to_mm

                if min_diameter_mm < diameter_in_mm < max_diameter_mm:
                    count += 1
                    cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                    cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                    cv2.putText(img, f"Diameter: {diameter_in_mm:.2f} mm",(a - r, b - r - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6, (255, 0, 0), 2)
            

            if 1 <= count <= 3:
                #print("HC-ONE DETECTED")
                return ["HC-ONE DETECTED",True]
            elif 4 <= count <= 5:
                #print("Piston DETECTED")
                return ["piston" , True]
            elif 7 <= count <= 13:
               # print("HC-THREE DETECTED")
               return ["HC-THRE" , True]
            elif count > 14:
                #print("NO COMPONENT DETECTION")
                return ["NO COMPONENT" , False]
        else:
            #print("NO COMPONENT DETECTION")
            return ["NO COMPONENT" , False]

        #cv2.imwrite("Detected Circle", img)
# cd = ComponentDetect("bluewasher/blue_washer.jpg")
# print(cd.DetectComponent())