"""Trueface SDK movement tracking demo"""
import os
from trueface.video import QVideoStream
import cv2
from trueface.object_detection import ObjectRecognizer
from trueface.recognition import FaceRecognizer
from trueface.tracking import COObjectTracker
from trueface.utils import iou


#init object recognizer
OBJECT_RECOGNITION = ObjectRecognizer(
    ctx='gpu',
    model_path="./tf-object_detection-yolov3/model.trueface",
    params_path="./tf-object_detection-yolov3/model.params",
    license=os.environ['TF_TOKEN'],
    classes="./tf-object_detection-yolov3/classes.names")


CAP = QVideoStream(src="./shopping.mp4").start()

#set track movements == to how many points to track, max steps == to how many points you wanna draw
OT = COObjectTracker(threshold=7, track_movements=20, max_steps=20)

cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def run_demo():
    """A movement tracking demo"""
    while True:
        #read frame and run face detect
        if CAP.stream.get(cv2.CAP_PROP_FRAME_COUNT) == CAP.stream.get(cv2.CAP_PROP_POS_FRAMES):
            break

        frame = CAP.read()

        result = OBJECT_RECOGNITION.predict(frame)

        OT.update_trackers(frame)
        OT.clean()

        for i, box in enumerate(result['boxes']):
            print("matched")
            #only track movement of persons
            if result['classes'][i] != "person":
                continue

            quality, matched_oid, tracked_bbox = OT.find_tracked_object(
                box,
                frame,
                method=0,
                track_movements=True)
            print(quality)

            #if tracked
            if matched_oid:
                # print("matched with tracker")
                (tleft, ttop, twidth, theight) = tracked_bbox
                OBJECT_RECOGNITION.draw_box(frame, (tleft, ttop, tleft+twidth, ttop+theight))
                OBJECT_RECOGNITION.draw_label(frame, (tleft, ttop), result['classes'][i])

            else:
                OT.track(box, frame, None)
                OBJECT_RECOGNITION.draw_box(frame, box)


        # for tracked_object_id in OT.tracked_objects:
        #     print(tracked_object_id)
        #     print(OT.tracked_objects[tracked_object_id]['movements'])
        #     print(len(OT.tracked_objects[tracked_object_id]['movements']))

        OT.draw_motion_tracks(frame)
        cv2.imshow('image', frame)
        if cv2.waitKey(33) == ord('q'):
            break

if __name__ == "__main__":
    run_demo()
