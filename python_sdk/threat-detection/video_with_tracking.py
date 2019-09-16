"""threat detection with tracking"""
import os
from trueface.threat_detection import ThreatDetection
from trueface.video import VideoStream
import tensorflow as tf
import cv2
from trueface.tracking import COObjectTracker
from trueface.recognition import BaseRecognizer


threshold = 0.9

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    threat_detection = ThreatDetection(
        "./threat-model/model.trueface",
        "./threat-model/model.params",
        "./threat-model/trueface-classes.csv",
        os.environ['TF_TOKEN'])

    cap = VideoStream(src=0).start()

    #create a tracker
    ot = COObjectTracker(
        threshold=10)

    base_r = BaseRecognizer(
        ctx='cpu',
        license=os.environ['TF_TOKEN'])

    counter = 0
    while True:
        frame = cap.read()
        if frame is None:
            print('image none')
            break

        ot.update_trackers(frame)
        ot.clean()

        if counter == 0  or counter % 30 == 0:

            prediction = threat_detection.predict(frame)
            print(prediction)
            if prediction:
                for pred in prediction:
                    if pred['score'] < threshold:
                        continue

                    (left, top, right, bottom) = pred['box']
                    quality, matched_oid, tracked_bbox = ot.find_tracked_object(
                        (left, top, right, bottom), frame)
                    
                    if not matched_oid:
                        ot.track((left, top, right, bottom), frame, pred['label'])

                    label = "{}: {:.2f}".format(pred['label'], pred['score'])
                    base_r.draw_box(
                        frame,
                        pred['box'])
                    base_r.draw_label(frame, (pred['box'][0], pred['box'][1] - 10), label)
        else:
            for obj in ot.tracked_objects:
                bbox = ot.tracked_objects[obj]['tracker'].get_position()
                x = int(bbox.left())
                y = int(bbox.top())
                w = int(bbox.width())
                h = int(bbox.height())
                base_r.draw_label(
                    frame,
                    (x, y),
                    ot.tracked_objects[obj]['identity'], 1, 2)
                base_r.draw_box(
                    frame,
                    [x, y, x+w, y+h])
        counter += 1

        cv2.imshow('Trueface.ai', frame)
        if cv2.waitKey(33) == ord('q'):
            break
