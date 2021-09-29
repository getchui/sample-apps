#!/usr/bin/env python3
# Sample code: Get frame from webcam, run face detection, blur all the detected faces
# Note: you will need to have the opencv-python module installed

import argparse
import tfsdk
import cv2
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_video", required=True, help="Input video filename")

opt = parser.parse_args()

options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
options.smallest_face_height = 10
# options.fd_mode = tfsdk.FACEDETECTIONMODE.VERSATILE
options.fd_filter = tfsdk.FACEDETECTIONFILTER.UNFILTERED
#options.fd_filter = tfsdk.FACEDETECTIONFILTER.HIGH_RECALL
# options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.FULL

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print("Invalid License Provided")
    print("Be sure to export your license token as TRUEFACE_TOKEN")
    quit()


# Use the default camera (TODO: Can change the camera source, for example to an RTSP stream)
cap = cv2.VideoCapture(opt.input_video)
if (cap.isOpened()== False):
    print("Error opening video stream")
    os._exit(1)


res, frame = cap.read()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out_width = width
out_height = height

fourcc = cv2.VideoWriter_fourcc(*'XVID')

if not isinstance(opt.input_video, int):
    out_directory = os.path.dirname(opt.input_video)
    filename = os.path.basename(opt.input_video)
else:
    out_directory = os.path.curdir
    filename = "camera{}.mkv".format(opt.input_video)


out = cv2.VideoWriter(os.path.join(out_directory, "redacted-{}".format(filename)),
                      fourcc, 25.0, (out_width, out_height))

while(True):
    # To skip some frames, uncomment the following
    # cap.grab()

    ret, frame = cap.read()
    if not ret:
        break
    if ret == False:
        continue

    # Set the image using the frame buffer. OpenCV stores images in BGR format
    res = sdk.set_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to set frame")
        continue

    # Run face detection
    face_box_and_landmarks = sdk.detect_faces()
    #face_box_and_landmarks =  list(filter(lambda face: face.score > 0.1, face_box_and_landmarks))

    img = frame.copy()

    # Blur all the faces
    blur = 80
    for face in face_box_and_landmarks:
        extra_padding = 50
        x1 = max([0, int(face.top_left.x-extra_padding)])
        x2 = min([width,int(face.bottom_right.x+extra_padding)])
        y1 = max([0,int(face.top_left.y-extra_padding)])
        y2 = min([height,int(face.bottom_right.y+extra_padding)])

        # Base the blur kernel size based on the face width
        face_width = x2 - x1

        img[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (blur, blur))

        # Draw the rectangle on the frame
        #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)


    # Display the resulting frame
    out.write(img)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

