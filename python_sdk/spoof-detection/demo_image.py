"""Demo with Tracking Processor"""
import os
import cv2
from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
from trueface.spoofv4 import SpoofDetector as SP4
import os
import sys

#init spoof detector
sp = SP4(
    './spoofv4/spoofv4.trueface',
    './spoofv4/spoofv4.params',
    os.environ['TF_TOKEN'],
    ctx='gpu')
threshold = 0.5

path = "test.jpg"

result = sp.predict(path, 0.5)
label = "Real:%0.2f Fake:%0.2f Pred: %s" % (
    result["real"], 
    result["fake"], 
    result["prediction"]) 

print(label)
