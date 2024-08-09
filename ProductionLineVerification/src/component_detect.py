import cv2 
from ultralytics import YOLO
class Detect():
    def __init__(self , image_path):
        self.image_path = image_path

    def ComponentDetction(self):
        model = YOLO('models/component.pt')
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
                orientation_labels = {0: 'HIDDEN_COVER_TWO'} 
                orientation_label = orientation_labels.get(class_id, 'Unknown')
                #print(orientation_labels.values())
            
                #print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
                return [orientation_label,True]
            else:
                return ["HIDDEN_COVER_TWO",False]
                #print("this is first else part")
                #print("this is ",orientation_labels.values())
                #return True
        #         #return None
        # else:
        #     return ["HIDDEN_COVER_TWO",False]
            #print("this is else part")
            #print(orientation_label.vales())
            # print("false")
            # return False
# det = Detect("resources/octogan.jpeg")
# print(det.ComponentDetction())