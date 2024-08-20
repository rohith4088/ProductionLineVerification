import cv2
import numpy as np

def detect_octagon(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # Check if the approximated contour is an octagon
        if len(approx) == 8:
            # Draw the octagon
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)

    cv2.imshow("Octagon Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = "images/current.jpg"
detect_octagon(image_path)
