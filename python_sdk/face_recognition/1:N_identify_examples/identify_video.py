from trueface.recognition import FaceRecognizer
from trueface.video import VideoStream
from trueface.face_attributes import FaceAttributes
import json
import cv2
import sys
import os
from datetime import datetime

if len(sys.argv) < 3:
    print("Usage: {} <path to video or camera id. 0 for first USB camera> <collection folder>".format(sys.argv[0]))
    sys.exit(1)

videofile = sys.argv[1]
if videofile.isdigit():
    videofile = int(videofile)

collection_folder = sys.argv[2]
collection_file = "{}.npz".format(collection_folder)

fr = FaceRecognizer(ctx='cpu',
                    fd_model_path='fd_model',
                    fr_model_path='model-lite/model.trueface',
                    params_path='model-lite/model.params',
                    license=os.environ['TOKEN'])

fr.create_collection(collection_folder,
                     collection_file, return_features=False)

cap = cv2.VideoCapture(videofile)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("detected_{}".format(videofile),fourcc, fps, (width,height))

i = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    if cv2.waitKey(18) & 0xFF == ord('q'):
        break
    bounding_boxes, points, chips = fr.find_faces(frame, return_chips=True,
                                                  return_binary=True)
    start_time = datetime.now()
    if bounding_boxes is None:
        out.write(frame)
        continue
    print("{} chips".format(len(chips)))
    for i, chip in enumerate(chips):
        identity = fr.identify(chip, threshold=0.3,
                               collection=collection_file)
        print(identity)
        if identity['predicted_label']:
            fr.draw_label(frame,
                          (int(bounding_boxes[i][0]),
                           int(bounding_boxes[i][1])),
                          identity['predicted_label'],
                          font_scale=1.5)

        fr.draw_box(frame, bounding_boxes[i])
    print("frame analysis took: {}".format(datetime.now()-start_time))
    out.write(frame)
    cv2.imshow('Trueface.ai', frame)


cap.release()
out.release()
cv2.destroyAllWindows()
