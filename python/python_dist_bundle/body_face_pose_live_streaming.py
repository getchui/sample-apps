# Sample code: Get frame from webcam, run face pose estimation and body pose estimation, draw results on the frame, display the frame
# Note: you will need to have the opencv-python module installed

import tfsdk
import cv2
import os
import math
import time
from colorama import Fore
from colorama import Style

def draw_body_pose(frame, body_landmarks):
    joint_pairs = [
        [0, 1], [1, 3], [0, 2], [2, 4],
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [12, 14], [13, 15], [14, 16]
    ]
    for i,j in enumerate(joint_pairs):
        Landmark1 = body_landmarks[0][joint_pairs[i][0]]
        Landmark2 = body_landmarks[0][joint_pairs[i][1]]
        if Landmark1.score < 0.2 or Landmark2.score < 0.2:
            continue

        cv2.line(frame, (int(Landmark1.point.x), int(Landmark1.point.y)), 
            (int(Landmark2.point.x), int(Landmark2.point.y)), (255, 0, 0), 2)

    for i,l in enumerate(body_landmarks[0]):
        if l.score < 0.2:
            continue
        cv2.circle(frame, (int(l.point.x), int(l.point.y)), 3, (0, 255, 0), -1);


def draw_pose_lines(yaw, pitch, roll, frame):

    #Center point for the axis we will draw
    origin = (100, 100);

    #Compute 3D rotation axis from yaw, pitch, roll
    #https://stackoverflow.com/a/32133715/4943329
    x1 = 100 * math.cos(yaw) * math.cos(roll);
    y1 = 100 * (math.cos(pitch) * math.sin(roll) + math.cos(roll) * math.sin(pitch) * math.sin(yaw));
    x2 = 100 * (-1 * math.cos(yaw) * math.sin(roll));
    y2 = 100 * (math.cos(pitch) * math.cos(roll) - math.sin(pitch) * math.sin(yaw) * math.sin(roll));
    x3 = 100 * math.sin(yaw);
    y3 = 100 * (-1 * math.cos(yaw) * math.sin(pitch));

    #Draw the arrows on the screen
    cv2.arrowedLine(frame, origin, (int(x1 + origin[0]), int(y1 + origin[1])), (255, 0, 0), 4, cv2.LINE_AA);
    cv2.arrowedLine(frame, origin, (int(x2 + origin[0]), int(y2 + origin[1])), (0, 255, 0), 4, cv2.LINE_AA);
    cv2.arrowedLine(frame, origin, (int(x3 + origin[0]), int(y3 + origin[1])), (0, 0, 255), 4, cv2.LINE_AA);
    return frame


options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
# options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
# options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.FULL

# Since we know we will use face detection and body pose estimation,
# we can choose to initialize these modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.face_detector = True
initializeModule.bodypose_estimator = True
options.initialize_module = initializeModule

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()


# Use the default camera (TODO: Can change the camera source, for example to an RTSP stream)
cap = cv2.VideoCapture(0)
if (cap.isOpened()== False): 
    print(f"{Fore.RED}Error opening video stream{Style.RESET_ALL}")
    os._exit(1)


while(True):
    # To skip some frames, uncomment the following
    # cap.grab()
    
    ret, frame = cap.read()
    if ret == False:
        continue

    # Set the image using the frame buffer. OpenCV stores images in BGR format
    res = sdk.set_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set frame.{Style.RESET_ALL}")
        continue

    # Detect the largest face, then compute the face pose
    found, faceBoxAndLandmarks = sdk.detect_largest_face()
    if found == True:
        res, yaw, pitch, roll = sdk.estimate_head_orientation(faceBoxAndLandmarks)
        if res == tfsdk.ERRORCODE.NO_ERROR:
            draw_pose_lines(yaw, pitch, roll, frame)

    # Run object detection, then run body pose estimation
    bounding_boxes = sdk.detect_objects()
    if (len(bounding_boxes) > 0) :
        body_landmarks = sdk.estimate_pose(bounding_boxes)
        if len(body_landmarks) > 0 :
            draw_body_pose(frame, body_landmarks)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




