import cv2
import time
import os
from pathlib import Path
from ProductionLineVerification.config import configuration
import requests
class CaptureSave():
    def __init__(self, images_folder="images", interval=3):
        self.cap = cv2.VideoCapture(0)
        self.images_folder = images_folder
        self.interval = interval
        if not os.path.exists(self.images_folder):
            os.makedirs(self.images_folder)

    def capture_and_save_frame(self, frame_count):
        ret, frame = self.cap.read()
        if ret:
            filename = f"{self.images_folder}/current.jpg"
            cv2.imwrite(filename, frame)
            url = ""
            with open(filename, 'rb') as file:
                try:
                    response = requests.post(url, files={'image': filename})
                    response.raise_for_status()  
                    print(f"Successfully uploaded {filename} to the server.")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to upload {filename}. Error: {e}") 
                    print(f"Saved frame {frame_count} to {filename}")
        else:
            print("Failed to capture frame")

    def get_latest_image_path(self):
        files = Path(self.images_folder).glob('*.jpg')
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
        return str(sorted_files[0]) if sorted_files else None
        #return f'{self.images_folder}/current.jpg'
    def process_latest_image(self):
        latest_image_path = self.get_latest_image_path()
        if latest_image_path:
            print(f"Latest image path: {latest_image_path}")
            bluob = configuration.BlueWasherDetect(latest_image_path)
            bluob.detect_washer()
            bluob.check_orientation()
            blackwh = configuration.blackWhiteDetect(latest_image_path)
            blackwh.BlackWhiteCheck()
        else:
            print("No images found in the specified folder.")

if __name__ == "__main__":
    capture_obj = CaptureSave()
    frame_count = 0
    while True:
        capture_obj.capture_and_save_frame(frame_count)
        frame_count += 1
        capture_obj.process_latest_image()
        time.sleep(capture_obj.interval)












# import cv2
# import time
# import os
# from pathlib import Path
# from ProductionLineVerification.config import configuration
# from ProductionLineVerification.src import capture_roi

# class CaptureSave():
#     def __init__(self, images_folder="images", interval=3):
#         self.cap = cv2.VideoCapture(0)
#         self.images_folder = images_folder
#         self.interval = interval
#         if not os.path.exists(self.images_folder):
#             os.makedirs(self.images_folder)

#     def capture_and_save_frame(self):
#         ret, frame = self.cap.read()
#         if ret:
#             filename = f"{self.images_folder}/current.jpg"
#             cv2.imwrite(filename, frame)
#             print(f"Saved frame to {filename}")
#         else:
#             print("Failed to capture frame")

#     def get_latest_image_path(self):
#         # Since we're overwriting the same file, the latest image path is simply the file name
#         return f"{self.images_folder}/current.jpg"

#     def process_latest_image(self):
#         latest_image_path = self.get_latest_image_path()
#         if os.path.exists(latest_image_path):
#             print(f"Processing latest image at {latest_image_path}")
#             try:
#                 # Attempt to read the image to ensure it's accessible
#                 img = cv2.imread(latest_image_path)
#                 if img is not None:
#                     bluob = configuration.BlueWasherDetect(latest_image_path)
#                     bluob.detect_washer(img)
#                     bluob.check_orientation(img)
#                     blackwh = configuration.blackWhiteDetect(latest_image_path)
#                     blackwh.BlackWhiteCheck(img)
#                 else:
#                     print("Failed to read the image file.")
#             except Exception as e:
#                 print(f"An error occurred while processing the image: {e}")
#         else:
#             print(f"No file found at {latest_image_path}")

# if __name__ == "__main__":
#     capture_obj = CaptureSave()
#     while True:
#         capture_obj.capture_and_save_frame()
#         capture_obj.process_latest_image()
#         time.sleep(capture_obj.interval)
