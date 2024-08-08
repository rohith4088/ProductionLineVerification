import cv2
from ultralytics import YOLO

class PistonDetect():
    def __init__(self , image_path):
        self.image_path = image_path
    def PistonCheck(self):
        """Checks black and white using a pre-trained YOLO model."""
        model = YOLO('models/piston.pt')
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"Failed to load image from {self.image_path}. Please check the file path.")
            return None
        results = model(img)
        if results and len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                box = result.boxes[0]
                print(box)
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                orientation_labels = {1:"piston"} 
                orientation_label = orientation_labels.get(class_id, 'Unknown')
            
                print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
                return orientation_label
            else:
                print("No piston detected.")
                return None
        else:
            print("No detections made.")
            return None

piston = PistonDetect("resources/piston.jpeg")
piston.PistonCheck()