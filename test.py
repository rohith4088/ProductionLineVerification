SEQ1 = False
SEQ2 = False
SEQ3 = False
COMP = False

import requests
import os 
from ProductionLineVerification.src import component_detect
from ProductionLineVerification.config import configuration
import cv2
from pathlib import Path
import time

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
            # url = ""
            # with open(filename, 'rb') as file:
            #     try:
            #         response = requests.post(url, files={'image': filename})
            #         response.raise_for_status()  
            #         print(f"Successfully uploaded {filename} to the server.")
            #     except requests.exceptions.RequestException as e:
            #         print(f"Failed to upload {filename}. Error: {e}") 
            #         print(f"Saved frame {frame_count} to {filename}")
        else:
            print("Failed to capture frame")

    def get_latest_image_path(self):
        files = Path(self.images_folder).glob('*.jpg')
        sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
        return str(sorted_files[0]) if sorted_files else None
        #return f'{self.images_folder}/current.jpg'
    def process_latest_image(self):
        if COMP == False:
            componentobject = component_detect.Detect("images/current.jpg")
            text = componentobject.ComponentDetction()
            print(text)
            url = "http://194.233.76.50:3004/uploadComp"
            try:
                response = requests.post(url, files={'result':text[0]})
                response.raise_for_status()
                print(f"Successfully uploaded {text[0]} to the server.")
            except requests.exceptions.RequestException as e:
                print(f"Failed to upload {text[0]}. Error: {e}") 
            if text[1] == True:
                COMP == True
        
        if COMP == True:

            if SEQ1 == False:
                blueobject = configuration.BlueWasherDetect("images/current.jpg")#image_path allocation pending
                detect_variable = blueobject.detect_washer()
                orientation_variable =blueobject.check_orientation()
                if detect_variable and orientation_variable:
                    SEQ1 == True
                    url = "http://194.233.76.50:3004/uploadSeq/blue"
                    try:
                        response = requests.post(url, files={'result':SEQ1})
                        response.raise_for_status()
                        print(f"Successfully uploaded {SEQ1} to the server.")
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to upload {SEQ1}. Error: {e}")
                # print(SEQ1 , SEQ2 , SEQ3)
            elif SEQ2 == False:
                yellowobject = configuration.YellowWasherDetect("images/current.jpg")
                detect_variable = yellowobject.detect_washer()
                if detect_variable:
                    SEQ2 == True
                    url = "http://194.233.76.50:3004/uploadSeq/yellow"
                    try:
                        response = requests.post(url, files={'result':SEQ2})
                        response.raise_for_status()
                        print(f"Successfully uploaded {SEQ2} to the server.")
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to upload {SEQ2}. Error: {e}")
                print("seq2")

            elif SEQ3 == False:
                bwobject = configuration.blackWhiteDetect("images/current.jpg")
                detect_variable = bwobject.BlackWhiteCheck()
                if detect_variable:
                    SEQ3== True
                    url = "http://194.233.76.50:3004/uploadSeq/bw"
                    try:
                        response = requests.post(url, files={'result':SEQ2})
                        response.raise_for_status()
                        print(f"Successfully uploaded {SEQ2} to the server.")
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to upload {SEQ2}. Error: {e}")
                print("seq3")

    if (SEQ1 and SEQ2 and SEQ3 and COMP):   
        SEQ1 = False
        SEQ2 = False
        SEQ3 = False
        COMP = False
    
if __name__ == "__main__":
    capture_obj = CaptureSave()
    frame_count = 0
    while True:
        capture_obj.capture_and_save_frame(frame_count)
        frame_count += 1
        capture_obj.process_latest_image()
        time.sleep(capture_obj.interval)