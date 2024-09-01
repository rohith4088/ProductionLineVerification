# import cv2
# from ultralytics import YOLO
# import numpy as np

# class CombinedDetector():
#     def __init__(self, image_path):
#         self.image_path = image_path
    
#     def DetectComponents(self):
#         # Load the image
#         img = cv2.imread(self.image_path)
#         if img is None:
#             print(f"Failed to load image from {self.image_path}. Please check the file path.")
#             return None
        
#         # Run inference on all models
#         octogon_result = self.OctogonDetection(img)
#         piston_result = self.PistonDetection(img)
#         hc_one_result = self.HCOneDetection(img)
        
#         # Determine the highest confidence component
#         max_confidence = max(octogon_result[0], piston_result[0], hc_one_result[0])
#         best_component = next((component for component, result in [('Octogon', octogon_result), ('PISTON', piston_result), ('HC_ONE', hc_one_result)] if result[0] == max_confidence))
        
#         return f"{best_component} DETECTED", True if max_confidence > 0 else False

#     def OctogonDetection(self, img):
#         # Convert image to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Apply CLAHE to enhance contrast
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         v = clahe.apply(gray)
#         pixel_to_mm = 1.0
#         min_diameter_mm = 10
#         max_diameter_mm = 40
        
#         # Detect circles
#         detected_circles = cv2.HoughCircles(v, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=40)
#         if detected_circles is not None:
#             detected_circles = np.uint16(np.around(detected_circles))
#             count = 0
#             for pt in detected_circles[0, :]:
#                 a, b, r = pt[0], pt[1], pt[2]
#                 diameter_in_mm = 2 * r * pixel_to_mm
#                 if min_diameter_mm < diameter_in_mm < max_diameter_mm:
#                     count += 1
#                     cv2.circle(img, (a, b), r, (0, 255, 0), 2)
#                     cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
#                     cv2.putText(img, f"Diameter: {diameter_in_mm:.2f} mm", (a - r, b - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
#             cv2.imwrite("detected_circles.jpg", img)

#             if 7 <= count <= 9:
#                 return ["HC-TWO DETECTED", True]

#     def PistonDetection(self, img):
#         model = YOLO('models/piston_final.pt')
#         results = model(img)
        
#         if not results or len(results) == 0:
#             return [0, False]
        
#         boxes = results[0].boxes
#         if not boxes:
#             return [0, False]
        
#         box = boxes[0]
#         class_id = int(box.cls[0])
#         confidence = float(box.conf[0])
        
#         return [confidence, True]

#     def HCOneDetection(self, img):
#         model = YOLO('models/hc_one.pt')
#         results = model(img)
        
#         if not results or len(results) == 0:
#             return [0, False]
        
#         boxes = results[0].boxes
#         if not boxes:
#             return [0, False]
        
#         box = boxes[0]
#         class_id = int(box.cls[0])
#         confidence = float(box.conf[0])
        
#         return [confidence, True]

# # Usage example
# detector = CombinedDetector("resources/piston.jpeg")
# result = detector.DetectComponents()
# print(result)
import cv2
from ultralytics import YOLO
import numpy as np

class CombinedDetector():
    def __init__(self, image_path):
        self.image_path = image_path
    
    def DetectComponents(self):
        # Load the image
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"Failed to load image from {self.image_path}. Please check the file path.")
            return None
        octogon_model = YOLO('models/component.pt')
        piston_model = YOLO('models/piston_final.pt')
        hc_one_model = YOLO('models/hc_one.pt')
        
        results = {
            'octogon': octogon_model(img),
            'piston': piston_model(img),
            'hc_one': hc_one_model(img)
        }
        octogon_result = self.ProcessOctogon(results['octogon'])
        piston_result = self.ProcessPiston(results['piston'])
        hc_one_result = self.ProcessHCOne(results['hc_one'])
        max_confidence = max(octogon_result[0], piston_result[0], hc_one_result[0])
        best_component = next((component for component, result in [('Octogon', octogon_result), ('PISTON', piston_result), ('HC_ONE', hc_one_result)] if result[0] == max_confidence))
        return f"{best_component} DETECTED", True if max_confidence > 0 else False

    @staticmethod
    def ProcessOctogon(result):
        if not result or len(result) == 0:
            return [0, False]
        
        boxes = result[0].boxes
        if not boxes:
            return [0, False]
        
        box = boxes[0]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        return [confidence, True]

    @staticmethod
    def ProcessPiston(result):
        if not result or len(result) == 0:
            return [0, False]
        
        boxes = result[0].boxes
        if not boxes:
            return [0, False]
        
        box = boxes[0]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        return [confidence, True]

    @staticmethod
    def ProcessHCOne(result):
        if not result or len(result) == 0:
            return [0, False]
        
        boxes = result[0].boxes
        if not boxes:
            return [0, False]
        
        box = boxes[0]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        return [confidence, True]

# Usage example
detector = CombinedDetector("resources/bluewrong2.jpeg")
result = detector.DetectComponents()
print(result)
