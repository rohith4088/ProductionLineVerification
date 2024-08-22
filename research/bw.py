import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_image(image_path):
    img = cv2.imread(image_path)

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding to separate the black component
    _, black_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Thresholding to separate the white ring
    _, white_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # Find contours of the black component
    black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours of the white ring
    white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualization
    debug_images = []
    debug_images.append(('Original', img))
    debug_images.append(('Grayscale', gray))
    debug_images.append(('Blurred', blurred))
    debug_images.append(('Black Threshold', black_thresh))
    debug_images.append(('White Threshold', white_thresh))

    result_img = img.copy()

    if black_contours and white_contours:
        # Get the largest black contour (the main component)
        black_component = max(black_contours, key=cv2.contourArea)

        # Get the largest white contour (should be the ring)
        white_ring = max(white_contours, key=cv2.contourArea)

        # Get bounding rectangles
        black_x, black_y, black_w, black_h = cv2.boundingRect(black_component)
        white_x, white_y, white_w, white_h = cv2.boundingRect(white_ring)

        # Draw contours and bounding boxes
        cv2.drawContours(result_img, [black_component], 0, (0, 255, 0), 2)
        cv2.drawContours(result_img, [white_ring], 0, (0, 0, 255), 2)
        cv2.rectangle(result_img, (black_x, black_y), (black_x + black_w, black_y + black_h), (255, 0, 0), 2)
        cv2.rectangle(result_img, (white_x, white_y), (white_x + white_w, white_y + white_h), (255, 255, 0), 2)

        debug_images.append(('Result', result_img))

        # Check if the white ring is on the right side of the black component
        if white_x > black_x + (black_w / 2):
            return "Correct", debug_images
        else:
            return "Incorrect", debug_images
    else:
        return "Black component or white ring not detected", debug_images

# Example usage
image_path = "resources/bw.jpeg"
result, debug_images = analyze_image(image_path)
print(result)

# Visualize debug images
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, (title, img) in enumerate(debug_images):
    if len(img.shape) == 2:
        axes[i].imshow(img, cmap='gray')
    else:
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(title)
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# import cv2
# import numpy as np

# def mask_shapes(image_path, min_width_black=0.05, min_length_white_black=10):
#     # Read the image
#     img = cv2.imread(image_path)

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Thresholding to separate the black component
#     _, black_thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

#     # Thresholding to separate the white component
#     _, white_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

#     # Find contours of the black component
#     black_contours, _ = cv2.findContours(black_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Find contours of the white component
#     white_contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter contours based on area (this is a heuristic; adjust as needed)
#     min_area = 100  # Minimum area to consider a contour significant

#     # Filter black contours
#     black_filtered = [cnt for cnt in black_contours if cv2.contourArea(cnt) > min_area]

#     # Filter white contours
#     white_filtered = [cnt for cnt in white_contours if cv2.contourArea(cnt) > min_area]

#     # Measure contour dimensions
#     for cnt in black_filtered:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = float(w)/h
#         if aspect_ratio >= 0.05/min_length_white_black:  # Check if width meets the criteria
#             cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

#     for cnt in white_filtered:
#         x, y, w, h = cv2.boundingRect(cnt)
#         aspect_ratio = float(w)/h
#         if aspect_ratio >= 0.05/min_length_white_black:  # Check if width meets the criteria
#             cv2.drawContours(img, [cnt], -1, (255, 0, 0), 2)

#     # Display the result
#     cv2.imshow("Filtered Image", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Example usage
# image_path = "resources/bw.jpeg"
# mask_shapes(image_path)




