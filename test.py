SEQ1 = False
SEQ2 = False
SEQ3 = False
import requests

from ProductionLineVerification.src import component_detect
from ProductionLineVerification.config import configuration
while True:
    componentobject = component_detect.Detect("images/current.jpg")
    text = componentobject.ComponentDetction()
    #print(text)
    url = "http://194.233.76.50:3004/uploadComp"
    try:
        response = requests.post(url, files={'result':text})
        response.raise_for_status()
        print(f"Successfully uploaded {text} to the server.")
    except requests.exceptions.RequestException as e:
        print(f"Failed to upload {text}. Error: {e}")
    break
while (SEQ1 == False or SEQ2 == False or SEQ3 == False):
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
        break
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
            url = "http://194.233.76.50:3004/uploadSeq/yellow"
            try:
                response = requests.post(url, files={'result':SEQ2})
                response.raise_for_status()
                print(f"Successfully uploaded {SEQ2} to the server.")
            except requests.exceptions.RequestException as e:
                print(f"Failed to upload {SEQ2}. Error: {e}")
        print("seq3")

    break