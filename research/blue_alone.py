import cv2
import numpy as np
import os
import requests

import cv2
import numpy as np
import os
import requests

def BlueWasherDetect(frame, roi_x, roi_y, roi_w, roi_h, lower, upper):
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    
    blurred = cv2.GaussianBlur(roi_frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow("Initial Mask", mask)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Apply Hough Circle Transform
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    # Adjusted Hough Circle Transform parameters
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=80, param2=40, minRadius=10, maxRadius=100)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
    
        for (x, y, radius) in circles:
        # Improved contour filtering: Check aspect ratio
            approx = cv2.approxPolyDP(np.array([[x-radius, y], [x+radius, y], [x, y-radius], [x, y+radius]]), 10, True)
            aspect_ratio = float(approx.shape[0]) / float(approx.shape[1])
        
            if cv2.contourArea(approx) > 500 and 0.7 < aspect_ratio < 1.3:  # Adjust aspect ratio thresholds as needed
            # Draw the circle and its center
                cv2.circle(frame, (x + roi_x, y + roi_y), radius, (0, 255, 0), 2)
                cv2.circle(frame, (x + roi_x, y + roi_y), 2, (0, 0, 255), -1)
            
                cv2.putText(frame, f"Detected Washer (r={radius})", (x + roi_x, y + roi_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
                return True, radius

    
    return False, None

# The rest of your code remains uncha
# def YellowWasherDetect(frame, roi_x, roi_y, roi_w, roi_h, lower, upper):
#     # Implementation similar to BlueWasherDetect
#     # Adjust color range and parameters for yellow detection
#     pass

def save_washer_image(frame, roi_x, roi_y, roi_w, roi_h, path):
    roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
    absolute_path = os.path.abspath(path)
    cv2.imwrite(absolute_path, roi_frame)
    assert os.path.exists(absolute_path), f"File does not exist: {absolute_path}"

def upload_image_to_api(text):
    url = "http://194.233.76.50:3003/uploadDummy"
    try:
        response = requests.post(url, files={'result': text})
        response.raise_for_status()
        print(f"Successfully uploaded {text} to the server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to upload {text}. Error: {e}")

# Main execution
cap = cv2.VideoCapture("WhatsApp Video 2024-06-26 at 15.17.39.mp4")
# cap = cv2.VideoCapture(0)  # Uncomment this line to use webcam

roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 300 
bluewasher_dir = 'bluewasher'
# yellowwasher_dir = 'yellowwasher'

# if not os.path.exists(yellowwasher_dir):
#     os.makedirs(yellowwasher_dir)
if not os.path.exists(bluewasher_dir):
    os.makedirs(bluewasher_dir)

blue_lower = np.array([90, 50, 50])
blue_upper = np.array([130, 255, 255])

#yellow_lower = np.array([20, 100, 100])
#yellow_upper = np.array([30, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
    
    #cv2.imshow("Original Frame", frame)
    
    blue_detected, blue_radius = BlueWasherDetect(frame, roi_x, roi_y, roi_w, roi_h, blue_lower, blue_upper)
    #yellow_detected, yellow_radius = YellowWasherDetect(frame, roi_x, roi_y, roi_w, roi_h, yellow_lower, yellow_upper)
    
    if blue_detected:
        print(f"Blue washer detected with radius {blue_radius}")
        file_name = f"bluewasher_{len(os.listdir(bluewasher_dir))}.jpg"
        full_path = os.path.join(bluewasher_dir, file_name)
        save_washer_image(frame, roi_x, roi_y, roi_w, roi_h, full_path)
        upload_image_to_api("DETECTED BLUE WASHER")
    
    
    if blue_detected:
        cv2.putText(frame, "Correct Sequence", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Incorrect Sequence", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Processed Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()