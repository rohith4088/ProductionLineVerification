from ProductionLineVerification.config import configuration
from ProductionLineVerification.src import capture_roi

bluob = configuration.BlueWasherDetect()
bluob.detect_washer()
bluob.check_orientation()

blackwh = configuration.blackWhiteDetect()
blackwh.BlackWhiteCheck()

# pistonobject = configuration.PistonDetect("resources/piston.jpeg")
# pistonobject.PistonCheck()
capture = capture_roi.CaptureSave()
capture.capture_and_save_frame()