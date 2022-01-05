# Sample code: Get frame from webcam, run face detection, then run blink detection on the largest face in the frame
# Note: you will need to have the opencv-python module installed

import tfsdk
import cv2
import os
import time
from colorama import Fore
from colorama import Style

def draw_label(image, point, label, color,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.8, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x_label, y_label = point
    cv2.rectangle(
        image,
        (x_label, y_label - size[1] - 10),
        (x_label + size[0], y_label),
        color,
        cv2.FILLED)

    cv2.putText(
        image, label.capitalize(), (x_label, y_label - 5), font, font_scale,
        (0, 0, 0), thickness, cv2.LINE_AA)


options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
options.smallest_face_height = 100 # We only want to run blink detection on large faces to ensure we accurately determine the state of the eyes

# options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
# options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.GPU_options = True

# Since we know we will use liveness,
# we can choose to initialize this modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.liveness = True
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
    print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
    quit()

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
    found, face_box_and_landmarks = sdk.detect_largest_face()

    if not found:
        # No face was found in the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Run blink detection
    res, blinkstate = sdk.detect_blink(face_box_and_landmarks)

    if res == tfsdk.ERRORCODE.EXTREME_FACE_ANGLE:
        # The head angle is too extreme
        draw_label(frame, (50, 50), "Face camera!", (255, 255, 255))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    elif res != tfsdk.ERRORCODE.NO_ERROR:
        # Not able to compute blink
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if blinkstate.is_left_eye_closed and blinkstate.is_right_eye_closed:
        # Both eyes are closed, it's a blink!
        draw_label(frame, (50, 50), "Blink", (0, 0, 255))

    else:
        # Both eyes are not closed, it's not a blink!
        draw_label(frame, (50, 50), "Open", (0, 255, 0))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




