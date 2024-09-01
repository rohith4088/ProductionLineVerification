import cv2
from ultralytics import YOLO

class HC_ONE():
    def __init__(self , image_path):
        self.image_path = image_path
    def HcCheck(self):
        model = YOLO('models/hc_one.pt')
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"Failed to load image from {self.image_path}. Please check the file path.")
            return None
        results = model(img)
        if results and len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                box = result.boxes[0]
                #print(box)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                orientation_labels = {0:"HC_ONE"} 
                hc_label = orientation_labels.get(class_id, 'Unknown')
                print(confidence)
                #print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
                #print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
                return hc_label == 'HC_ONE'
            else:
                return False
        else:
            return False
hc = HC_ONE("resources/hc_one.jpeg")
print(hc.HcCheck())