from ProductionLineVerification.config import configuration
from ProductionLineVerification.src import component_detect
import os
from pathlib import Path
# import logging
#from schedule import run_all , repeat , run_pending , every

# bluob = configuration.BlueWasherDetect()
# blackwh = configuration.blackWhiteDetect()
capture = component_detect.CaptureSave()

def get_latest_image_path(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")
    
    files = Path(folder_path).glob('*.jpg') 
    sorted_files = sorted(files, key=os.path.getmtime, reverse=True)
    return str(sorted_files[0]) if sorted_files else None

capture.capture_and_save_frame()
latest_image_path = get_latest_image_path("images")
if latest_image_path:
    print(f"Latest image path: {latest_image_path}")
    bluob = configuration.BlueWasherDetect(latest_image_path)
    bluob.detect_washer()
    bluob.check_orientation()
    blackwh = configuration.blackWhiteDetect(latest_image_path)
    capture = component_detect.CaptureSave()
    # bluob.detect_washer()
    # bluob.check_orientation()
    blackwh.BlackWhiteCheck()
else:
    print("No images found in the specified folder.")
