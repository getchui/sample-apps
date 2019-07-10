from trueface.motion import MotionDetector
from trueface.video import VideoStream
import cv2

#please use 0.6.2 if you desire

cap = VideoStream(src="rtsp://192.168.1.11:554/11").start()

# apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
# motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
# frames count allows you to pick the count of frames to track heatmaps

motion = MotionDetector(cap.read(), threshold=1, max_value=3, frames=100)

count = 0
total = 1000
while True:
    frame = cap.read()
    frame = motion.detect(frame)
    count += 1
    #motion.fade(ratio=0.75) fades motion, use if not use frames= param
    cv2.imshow("image", frame)
    if cv2.waitKey(33) == ord('q'):
        break