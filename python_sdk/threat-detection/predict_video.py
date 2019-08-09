from trueface.threat_detection import ThreatDetection
from trueface.video import VideoStream
import tensorflow as tf
import cv2
import os

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    threat_detection = ThreatDetection(
        "./threat-model/model.trueface", 
        "./threat-model/model.params", 
        "./threat-model/trueface-classes.csv", 
        os.environ['TF_TOKEN'])

    cap = VideoStream(src=0).start()

    while True:
        frame = cap.read()
        if frame is None:
            print('image none')
            break

        prediction = threat_detection.predict(frame)
        print(prediction)
        if prediction:

            for pred in prediction:

                if pred['score'] < 0.5:
                    continue

                label = "{}: {:.2f}".format(pred['label'], pred['score'])
                cv2.rectangle(frame, (pred['box'][0], pred['box'][1]), (pred['box'][2], pred['box'][3]),
                    (0, 255, 0), 2)
                cv2.putText(frame, label, (pred['box'][0], pred['box'][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Trueface.ai', frame)
        if cv2.waitKey(33) == ord('q'):
            break
