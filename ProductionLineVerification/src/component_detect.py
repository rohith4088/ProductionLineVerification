# import cv2
# import time
# import os

# cap = cv2.VideoCapture(0)
# class CaptureSave():
#     def capture_and_save_frame(frame_count):
#         ret, frame = cap.read()
#         if ret:
#             filename = f"images/frame{frame_count}.jpg"
#             cv2.imwrite(filename, frame)
#             print(f"Saved frame {frame_count} to {filename}")
#         else:
#             print("Failed to capture frame")

#     if __name__ == "__main__":
#         images_folder = "images"
#         interval = 3 
#         if not os.path.exists(images_folder):
#             os.makedirs(images_folder)
    
#         frame_count = 0
#         while True:
#             capture_and_save_frame(frame_count)
#             frame_count += 1
#             time.sleep(interval)

# # obj = CaptureSave()
# # obj.capture_and_save_frame()

class Detect():
    def __init__(self , image_path):
        self.image_path = image_path
    
    def ComponentDetction(self):
        component_name = "dummy"
        return component_name