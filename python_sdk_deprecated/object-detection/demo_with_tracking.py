"""Demo with Tracking Processor"""
import os
import cv2
from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
from trueface.server import create_server
from trueface.object_detection import ObjectRecognizer
from trueface.tracking import COObjectTracker



#init object recognizer
object_recognition = ObjectRecognizer(ctx='gpu',
                      model_path="./tf-object_detection-mobilenet/model.trueface",
                      params_path="./tf-object_detection-mobilenet/model.params",
                      license=os.environ['TF_TOKEN'],
                      classes="./tf-object_detection-mobilenet/classes.names")

#initialize video capture from your webcam
cap = VideoStream(src=0).start()

#create a tracker
ot = COObjectTracker(
    threshold=10)

#start streaming server
counter = 0
while(True):
    frame = cap.read()
    ot.update_trackers(frame)
    ot.clean()
    #if frame 0 or every 30 frames
    if counter == 0  or counter % 30 == 0:
        result = object_recognition.predict(frame)
        print(result)
        for i, box in enumerate(result['boxes']):
            (left, top, right, bottom) = box
            quality, matched_oid, tracked_bbox = ot.find_tracked_object((left, top, right, bottom), frame)
            if not matched_oid:
                ot.track((left, top, right, bottom), frame, result['classes'][i])
            object_recognition.draw_label(frame, (int(box[0]), int(box[1])), result['classes'][i])
            object_recognition.draw_box(frame, box)

    else:
        for obj in ot.tracked_objects:
            bbox = ot.tracked_objects[obj]['tracker'].get_position()
            x = int(bbox.left())
            y = int(bbox.top())
            w = int(bbox.width())
            h = int(bbox.height())
            object_recognition.draw_label(frame, (x, y),
                ot.tracked_objects[obj]['identity'], 1, 2)
            object_recognition.draw_box(frame,
                [x, y, x+w, y+h])
    counter += 1
    cv2.imshow('Trueface.ai', frame)
    if cv2.waitKey(33) == ord('q'):
        break
