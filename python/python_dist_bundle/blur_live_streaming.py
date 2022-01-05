# Sample code: Get frame from webcam, run face detection, blur all the detected faces
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

# Since we know we will use face detection,
# we can choose to initialize this modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.face_detector = True
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
        print(f"{Fore.RED}Unable to set frame{Style.RESET_ALL}")
        continue

    # Run face detection
    face_box_and_landmarks = sdk.detect_faces()

    img = frame.copy()

    # Blur all the faces
    for face in face_box_and_landmarks:
        x1 = int(face.top_left.x)
        x2 = int(face.bottom_right.x)
        y1 = int(face.top_left.y)
        y2 = int(face.bottom_right.y)

        # Base the blur kernel size based on the face width
        face_width = x2 - x1
        blur = int(face_width / 8)

        img[y1:y2, x1:x2] = cv2.blur(frame[y1:y2, x1:x2], (blur, blur))
        
        # Draw the rectangle on the frame
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




