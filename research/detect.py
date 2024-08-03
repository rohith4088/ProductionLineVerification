import cv2
import numpy as np

def detect_washer(image_path, color_name='blue', lower=np.array([20, 80, 50]), upper=np.array([180, 255, 255])):
    # Load the image
    frame = cv2.imread(image_path)
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a binary mask where white represents the color within the specified range
    mask = cv2.inRange(hsv, lower, upper)
    
    # Optional: Enhance edge detection
    # mask = cv2.GaussianBlur(mask, (5, 5), 0)
    # mask = cv2.Canny(mask, 50, 150)
    
    # Display the mask to visually inspect the color filtering
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Find circles in the mask
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=15, param1=50, param2=30, minRadius=10, maxRadius=200)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        
        # Additional processing can be added here if needed
        
        return True , circles[0]
    else:
        return False, None
    

# Example usage
image_path = "/Users/rohithr/Desktop/wipro_clean/bluewasher/blue_washer.jpg"
detected, circles = detect_washer(image_path)

if detected:
    print("Blue Washer detected.")
    print("Number of detected circles:", len(circles))
else:
    print("No washer detected.")
