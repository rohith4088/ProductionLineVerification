import requests
import os 
from ProductionLineVerification.src import component_detect
from ProductionLineVerification.config import configuration
import cv2
from pathlib import Path
import time


#global SEQ1;global SEQ2;global SEQ3;global COMP 
# global COMP  
# global SEQ3 
# global SEQ2
# global SEQ1
SEQ1 = False
SEQ2 = False
SEQ3 = False
COMP = False

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
            x, y, w, h = 100, 100, 300, 300
            roi = frame[y:y+h, x:x+w]
            filename = f"{self.images_folder}/current.jpg"
            cv2.imwrite(filename, roi)
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
        global COMP
        latest_image_path = self.get_latest_image_path()
        # print("--------------")
        # print(latest_image_path)
        # print("----------")
        # print(COMP)
        if  COMP == False:
            componentobject = component_detect.ComponentDetect(latest_image_path)
            text = componentobject.DetectComponent()
            # print("-------")
            # print(text)
            # print("--------")
            # print("componanent_name",text[0])
            if text[1] == True:
                url = "http://194.233.76.50:3004/uploadComp/"+text[0]
                try:
                    response = requests.post(url)
                    response.raise_for_status()
                    print(f"Successfully uploaded {text[0]} to the server.")
                except requests.exceptions.RequestException as e:
                    print(f"Failed to upload {text[0]}. Error: {e}") 
                COMP = True
                #print(COMP)
        #print(COMP)
        if COMP == True:
            global SEQ1 , SEQ2 , SEQ3
            if SEQ1 == False:
                print("entering blue sequence")
                latest_image_path = self.get_latest_image_path()
                # print("---------")
                # print(latest_image_path)
                blueobject = configuration.BlueWasherDetect(latest_image_path)#image_path allocation pending
                detect_variable = blueobject.detect_washer()
                # print("-----------")
                # print("detect_variable",detect_variable)
                orientation_variable =blueobject.check_orientation()
                # print("-----------")
                # print("orreintation_varible",orientation_variable)
                if detect_variable and orientation_variable:
                    SEQ1 = True
                    url = "http://194.233.76.50:3004/uploadSeq/blue"
                    try:
                        response = requests.post(url, files={'result':SEQ1})
                        response.raise_for_status()
                        print(f"Successfully uploaded {SEQ1} to the server.")
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to upload {SEQ1}. Error: {e}")
                # print(SEQ1 , SEQ2 , SEQ3)
            elif SEQ2 == False:
                print("entering yellow sequence")
                latest_image_path = self.get_latest_image_path()
                yellowobject = configuration.YellowWasherDetect(latest_image_path)
                detect_variable = yellowobject.detect_washer()
                if detect_variable:
                    SEQ2 = True
                    url = "http://194.233.76.50:3004/uploadSeq/yellow"
                    try:
                        response = requests.post(url, files={'result':SEQ2})
                        response.raise_for_status()
                        print(f"Successfully uploaded {SEQ2} to the server.")
                    except requests.exceptions.RequestException as e:
                        print(f"Failed to upload {SEQ2}. Error: {e}")
                print("seq2")

            elif SEQ3 == False:
                print("enetring black and white sequence")
                latest_image_path = self.get_latest_image_path()
                bwobject = configuration.blackWhiteDetect(latest_image_path)
                detect_variable = bwobject.BlackWhiteCheck()
                if detect_variable:
                    SEQ3 = True
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