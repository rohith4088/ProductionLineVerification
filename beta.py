import cv2
import numpy as np
import os
import requests

"""
Tunable Parameter List:
--> radius : minRadius and maxRadius (suggested not to change)
--> ROI : roi_x, roi_y, roi_w, roi_h = 100,100,300,300 (DEAFULT VALUE)
--> color_ranges
"""


"""
TO-DO'S
->problem : the blue is getting recognised properly but the yellow washer is also getting recognised in the blue washers mask
->possible solution : maybe something to do with the HSV range
->problem : A trailing problem to this is the yellow washer snaps are getting stored in both blue washer dir and as well as yellow washer dir.
->problem : automatic trigger of the program (once the octagonal component is detected)
->solution : the octagonal script is added , just have to integrate
->probelm : triggering the black and white washer detection YOLO model
->solution : should find a way to trigger the model when the component is flipped , maybe trace the geometry of the flipped component
->MAIN PROBELM : THE SEQUENCE DETECTION IS NOT HAPPENING BECCUSE OF THE OVERLAYED MASK EXPOSURE IN THE BLUE AND YELLOW REGION
->walk_around for detection yellow washer is to take the orientation check as a secondary parameter to trigger the yellow washer function,
logically the yellow washer follows the blue washer , hence this works but is heavily dependnt on the color(which is not reccomended)
"""


def detect_washer(frame, roi_x, roi_y, roi_w, roi_h, color_name, lower, upper):
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("mask" , mask)
    # kernel = np.ones((5,5), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100) #The radius can be adjusted
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # for i in circles[0, :]:
        #     cv2.circle(frame, (i[0] + roi_x, i[1] + roi_y), i[2], (0, 255, 0), 2) #consider these as the debugging steps you can disable them if you want.
        #     cv2.circle(frame, (i[0] + roi_x, i[1] + roi_y), 2, (0, 0, 255), 3)
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area > 500:

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x + roi_x, y + roi_y), (x+w + roi_x, y+h + roi_y), (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x + roi_x, y + roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            #area = cv2.contourArea(cnt)
                radius = int(np.sqrt(area / np.pi))
                return True, radius
            else:
                return False, None
    else:
        return False, None
    # roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()  # Extract the ROI
    # blurred_roi = cv2.GaussianBlur(roi_frame, (5, 5), 0)  # Apply Gaussian blur
    # hsv = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow("mask", mask)
    
    # circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     for i in circles[0, :]:
    #         cv2.circle(frame, (i[0] + roi_x, i[1] + roi_y), i[2], (0, 255, 0), 2)
    #         cv2.circle(frame, (i[0] + roi_x, i[1] + roi_y), 2, (0, 0, 255), 3)
        
    #     contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     if len(contours) > 0:
    #         cnt = max(contours, key=cv2.contourArea)
    #         area = cv2.contourArea(cnt)
    #         if area > 500:
    #             x, y, w, h = cv2.boundingRect(cnt)
    #             cv2.rectangle(frame, (x + roi_x, y + roi_y), (x+w + roi_x, y+h + roi_y), (0, 255, 0), 2)
    #             cv2.putText(frame, color_name, (x + roi_x, y + roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    #             radius = int(np.sqrt(area / np.pi))
    #             return True, radius
    #         else:
    #             return False, None
    # else:
    #     return False, None


from ultralytics import YOLO

def BlackWhiteCheck(image_path):
    """Checks black and white using a pre-trained YOLO model."""
    model = YOLO('blackwhite.pt')
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image from {image_path}. Please check the file path.")
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

# Initialize video capture object for the webcam
# cap = cv2.VideoCapture("/Users/rohithr/Desktop/wipro/WhatsApp Video 2024-06-26 at 15.17.39.mp4")
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()



import cv2
import os

def save_blue_washer_image(frame, roi_x, roi_y, roi_w, roi_h, base_directory="bluewasher"):
    """Saves a cropped image of the blue washer region, overwriting any existing image."""
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    output_filename = os.path.join(base_directory, "blue_washer.jpg")
    cv2.imwrite(output_filename, roi_frame)
    print(f"Blue washer image saved as {output_filename}")

def save_yellow_washer_image(frame, roi_x, roi_y, roi_w, roi_h, base_directory="yellowwasher"):
    """Saves a cropped image of the yellow washer, overwriting any existing image."""
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    output_filename = os.path.join(base_directory, "yellow_washer.jpg")
    cv2.imwrite(output_filename, roi_frame)
    print(f"Yellow washer image saved as {output_filename}")

def check_orientation(image_path):
    """only checks for the blue washer as of now"""
    frame = cv2.imread(image_path)
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
            print("Downside")
        else:
            print("Upside")
    else:
        print("Detected yellow washer")
        file_name = f'yellowwasher_{len(os.listdir(yellowasher_dir))}.jpg'
        full_path = os.path.join(yellowasher_dir, file_name)
        save_yellow_washer_image(frame, roi_x, roi_y, roi_w, roi_h, full_path)
        upload_image_to_api("DETECTED YELLOW WASHER")

def upload_image_to_api(text):
    url = "http://194.233.76.50:3003/uploadDummy"
    try:
        response = requests.post(url, files={'result': text})
        response.raise_for_status()
        print(f"Successfully uploaded {text} to the server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to upload {text}. Error: {e}")

cap = cv2.VideoCapture("resources/WhatsApp Video 2024-06-26 at 15.17.39.mp4")
#cap = cv2.VideoCapture(0)

roi_x, roi_y, roi_w, roi_h = 100,100,300,300 
bluewasher_dir = 'bluewasher'
yellowasher_dir = 'yellowasher'
if not os.path.exists(yellowasher_dir):
    os.makedirs(yellowasher_dir)
if not os.path.exists(bluewasher_dir):
    os.makedirs(bluewasher_dir)
washer_sequence = ['blue']#]#, 'black']#, 'white']
color_ranges = {
    'blue': (np.array([20, 80, 50]), np.array([180, 255, 255])),
    'yellow': (np.array([15, 150, 150]), np.array([35, 255, 255])),}
#     # 'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
#     # 'white': (np.array([0, 0, 200]), np.array([180, 20, 255]))
# } #color ranges are too narrow
# color_ranges = {
#     'blue': (np.array([20, 80, 50]), np.array([180, 255, 255])),
#     'yellow': (np.array([15, 150, 150]), np.array([35, 255, 255])),}
#     'black': (np.array([0, 0, 0]), np.array([180, 255, 30]))
#     #'white': (np.array([0, 0, 200]), np.array([180, 30, 255]))
# }

while True:
    ret, frame = cap.read()
    if not ret:
        break  
    
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    
    detected_sequence = []
    for color_name in washer_sequence:
        lower, upper = color_ranges[color_name]
        detected, radius = detect_washer(frame, roi_x, roi_y, roi_w, roi_h, color_name, lower, upper)
    
        if detected:
            detected_sequence.append(color_name)
            print(f"Detected {color_name} washer with radius {radius}")
        
            if color_name == 'blue':
                file_name = f"blue_washer.jpg"
                full_path = os.path.join(bluewasher_dir, file_name)
                save_blue_washer_image(frame, roi_x, roi_y, roi_w, roi_h)
                upload_image_to_api("DETECTED BLUE WASHER")
                check_orientation(full_path)
            # elif color_name == 'yellow':
            #     file_name = f'yellowwasher_{len(os.listdir(yellowasher_dir))}.jpg'
            #     full_path = os.path.join(yellowasher_dir, file_name)
            #     save_yellow_washer(frame, roi_x, roi_y, roi_w, roi_h, full_path)
            #     upload_image_to_api("DETECTED YELLOW WASHER")
            else:
                break  
    if detected_sequence == washer_sequence:
        cv2.putText(frame, "Correct Sequence", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Incorrect Sequence", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Washer Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
