import cv2
import numpy as np

# Initialize video capture object
#cap = cv2.VideoCapture("resources/assembly_video.mp4")
cap = cv2.VideoCapture(0)
cap.set(1920 , 1080)

# Define color ranges in HSV format
color_ranges = {
    'blue': (np.array([94, 80, 2]), np.array([150, 255, 255])),
    'yellow': (np.array([15, 150, 150]), np.array([35, 255, 255])),
    'black': (np.array([0, 0, 0]), np.array([180, 255, 30])),
    'white': (np.array([0, 0, 200]), np.array([180, 20, 255]))
}


washer_sequence = ['blue', 'yellow', 'black', 'white']
# for colorname in washer_sequence:
#     lower , upper = color_ranges[colorname]
#     print(lower, upper)
roi_x, roi_y, roi_w, roi_h = 100, 100, 260, 240  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].copy()
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Example for blue color detection
    lower_blue, upper_blue = np.array([94, 80, 2]), np.array([130, 255, 255])
    lower_yellow , upper_yellow = np.array([15, 150, 50]), np.array([35, 255, 255])
    lower_black , upper_black = np.array([0, 0, 0]), np.array([180, 255, 30])
    lower_white , upper_white = np.array([0, 0, 200]), np.array([180, 20, 255]) 
    blue = cv2.inRange(hsv, lower_blue, upper_blue)
    yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    black = cv2.inRange(hsv, lower_black, upper_black)
    white = cv2.inRange(hsv, lower_white, upper_white)
    cv2.imshow("blue", blue)
    # cv2.imshow("yellow", yellow)
    # cv2.imshow("black", black)
    # cv2.imshow("white", white)
    # cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
