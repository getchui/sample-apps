"""Producer example, runs on live stream extracting faces and pushes to queue for processing"""
from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from multiprocessing import Pool, Process
import time
import signal
from trueface.utils import RedisQueue
import base64
import json
import os


face_detector = FaceRecognizer(ctx='gpu',
               fd_model_path='./fd_model',
               license=os.environ['TF_TOKEN'])

q = RedisQueue('office_camera')

vcap = VideoStream(src=0).start()


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def detect(frame):
    bounding_boxes, points, chips = face_detector.find_faces(frame, return_chips=True, return_binary=True)
    return bounding_boxes, points, chips

p = Pool(1, init_worker)

frames = []
counter = 0
while True:
    #collects 100 frames then use pool.map with the face detector to parallelize face extraction
    if len(frames) == 100:
        try:
            start_time = time.time()
            detections = p.map(detect, frames)
            print(time.time() - start_time)
            allchips = []
            allbounding_boxes = []
            for detection in detections:
                bounding_boxes = detection[0]
                points = detection[1]
                chips = detection[2]
                allbounding_boxes.extend(bounding_boxes)
                allchips.extend(chips)

            #debug info
            print('bounding boxes', len(allbounding_boxes))
            print('chips', len(allchips))
            print(np.asarray(allchips).shape)

            #reset frames array
            frames = []

            data = {
                "chips":base64.b64encode(np.asarray(allchips)),
                "chip_count":len(allchips)
            }

            q.put(json.dumps(data))
            counter += 1
            print('batch', counter)
        except Exception as error:
            print(error)
            p.terminate()
            p.join()
    frame = vcap.read()
    frames.append(frame)
