from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream, QVideoStream
import cv2
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)
from multiprocessing import Pool, Process
from multiprocessing.pool import ThreadPool
from threading import Thread
import time
import signal
from trueface.utils import RedisQueue
import base64
import numpy as np
import cv2
import os
import json

face_recognition = FaceRecognizer(ctx='gpu',
               fd_model_path='/home/nchafni/Development/models/fd_model',
               fr_model_path='/home/nchafni/Development/models/model-tfv2/model.trueface', 
               params_path='/home/nchafni/Development/models/model-tfv2/model.params',
               license=os.environ['TF_TOKEN'])

q = RedisQueue('office_camera')

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
                        collection='./trueface_collection.npz',
                        batch_size=24,
                        threshold=0.3)

        print(result)