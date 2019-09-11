from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
import cv2
import numpy as np
from fd import preform_face_detect
import logging
logging.basicConfig(level=logging.DEBUG)
from multiprocessing import Pool, Process
from multiprocessing.pool import ThreadPool
from threading import Thread
import time
import signal
from queue import RedisQueue
import base64
import numpy as np
import cv2
import os
import json

fr = FaceRecognizer(ctx='gpu',
               fd_model_path='../models/fd_model',
               fr_model_path='../models/model-tfv2/model.trueface', 
               params_path='../models/model-tfv2/model.params',
               license=os.environ['TF_TOKEN'])

q = RedisQueue('office1')

#p = ThreadPool(5)

frames = []
counter = 0
while True:
    data = json.loads(q.get())
    batch = np.frombuffer(base64.b64decode(data['chips']), dtype=np.uint8)
    if batch is not None:
        batch = batch.reshape((data['chip_count'], 112, 112, 3))
        #imgs = [cv2.imdecode(image, 1) for image in batch]
        #print len(fr.batch_get_features(batch))
        result = face_recognition.batch_identify(
                        batch,
                        db=self.dbservice,
                        collection=self.config['fr_collection'],
                        batch_size=self.config['fr_batch_size'],
                        threshold=self.config['face_recognition_threshold'])

        print(results)