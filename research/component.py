import cv2
from ultralytics import YOLO

class blackWhiteDetect():
    def __init__(self , image_path):
        self.image_path = image_path
    # file3 = open("black.log" , 'w')

    # @profile(stream = file3)
    def BlackWhiteCheck(self):
        """Checks black and white using a pre-trained YOLO model."""
        model = YOLO('models/component.pt')
        img = cv2.imread(self.image_path)
        if img is None:
            print(f"Failed to load image from {self.image_path}. Please check the file path.")
            return None
        results = model(img,show = True)
        if results and len(results) > 0:
            result = results[0]
            if len(result.boxes) > 0:
                box = result.boxes[0]
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                orientation_labels = {0:'HIDDEN_COVER_TWO'} 
                orientation_label = orientation_labels.get(class_id, 'Unknown')
            
                #print(f"Washer orientation: {orientation_label} (Confidence: {confidence:.2f})")
                #return orientation_label
                return [orientation_label , True]
            else:
                return [orientation_label , False]

                #return None
        # else:
        #     #print("false")
        #     return [orientation_label,False]
        #     #return None


obj = blackWhiteDetect('resources/octogan.jpeg')
print(obj.BlackWhiteCheck())