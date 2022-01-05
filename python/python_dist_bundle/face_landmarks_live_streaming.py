# Sample code: Get frame from webcam, run face detection, then get the 106 facial landmarks for all the detected faces.
# Note: you will need to have the opencv-python module installed

import tfsdk
import cv2
import os
import time
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
# options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
# options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.GPU_options = True

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

    # Run face detection
    face_box_and_landmarks = sdk.detect_faces()

    for fb in face_box_and_landmarks:
        # Detect the 106 facial landmarks
        res, landmarks = sdk.get_face_landmarks(fb)
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print(f"{Fore.RED}Unable to get face landmarks{Style.RESET_ALL}")
            continue

        # Draw the 106 facial landmark points over the face in red
        for landmark in landmarks:
            frame = cv2.circle(frame, (int(landmark.x), int(landmark.y)), radius=1, color=(0, 0, 255), thickness=2 )

        # To also draw the 5 facial landmarks in blue, uncomment the following code:
        # for landmark in fb.landmarks:
            # frame = cv2.circle(frame, (int(landmark.x), int(landmark.y)), radius=1, color=(255, 0, 0), thickness=2 )            


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




