"""Demo with Tracking Processor"""
import os
import cv2
from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
from trueface.spoofv4 import SpoofDetector as SP4
import os

#init spoof detector
sp = SP4(
    './spoofv4/spoofv4.trueface',
    './spoofv4/spoofv4.params',
    os.environ['TF_TOKEN'],
    ctx='gpu')
threshold = 0.5

#initialize video capture from your webcam
cap = VideoStream(src=0).start()

counter = 0
while(True):
    frame = cap.read()
    result = sp.predict(frame, 0.5)
    print(result)
    counter += 1
    label = "Real:%0.2f Fake:%0.2f Pred: %s" % (
        result["real"], 
        result["fake"], 
        result["prediction"]) 

    sp.draw_label(frame, 
         (int(20), 
          int(20)), 
         label)
    cv2.imshow('Trueface.ai', frame)
    if cv2.waitKey(33) == ord('q'):
        break
