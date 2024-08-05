from ProductionLineVerification.config import configuration
#from ProductionLineVerification.src import capture_roi


bluob = configuration.BlueWasherDetect("bluewasher/blue_washer.jpg")
bluob.detect_washer()
bluob.check_orientation()

#blackwh = configuration.blackWhiteDetect("resources/bw.jpeg")
