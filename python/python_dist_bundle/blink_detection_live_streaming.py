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


# Start by specifying the configuration options to be used.
# Can choose to use the default configuration options if preferred by calling the default SDK constructor.
# Learn more about the configuration options: https://reference.trueface.ai/cpp/dev/latest/py/general.html
options = tfsdk.ConfigurationOptions()
# The face recognition model to use. TFV5_2 balances speed and accuracy.
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5_2
# The object detection model to use.
options.obj_model = tfsdk.OBJECTDETECTIONMODEL.ACCURATE
# The face detection filter.
options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
# The face detection model
options.fd_model = tfsdk.FACEDETECTIONMODEL.FAST
# Smallest face height in pixels for the face detector.
options.smallest_face_height = 100 # Only want to run blink detection on larger faces to ensure we get accurate results
# The path specifying the directory containing the model files which were downloaded.
options.models_path = os.getenv('MODELS_PATH') or './'
# Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
options.fr_vector_compression = False
# Database management system for the storage of biometric templates for 1 to N identification.
options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE

# Encrypt the biometric templates stored in the database
# If encryption is enabled, must provide an encryption key
options.encrypt_database.enable_encryption = False
options.encrypt_database.key = "TODO: Your encryption key here"

# Initialize module in SDK constructor.
# By default, the SDK uses lazy initialization, meaning modules are only initialized when they are first used (on first inference).
# This is done so that modules which are not used do not load their models into memory, and hence do not utilize memory.
# The downside to this is that the first inference will be much slower as the model file is being decrypted and loaded into memory.
# Therefore, if you know you will use a module, choose to pre-initialize the module, which reads the model file into memory in the SDK constructor.
options.initialize_module.blink_detector = True

# Options for enabling GPU
# We will disable GPU inference, but you can easily enable it by modifying the following options
# Note, you may require a specific GPU enabled token in order to enable GPU inference.
options.GPU_options = False # TODO: Change this to true to enable GPU
options.GPU_options.device_index = 0;

gpuModuleOptions = tfsdk.GPUModuleOptions()
gpuModuleOptions.max_batch_size = 4
gpuModuleOptions.opt_batch_size = 1
gpuModuleOptions.max_workspace_size = 2000
gpuModuleOptions.precision = tfsdk.PRECISION.FP16

# Note, you can set separate GPU options for each GPU supported module
options.GPU_options.blink_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_blur_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_landmark_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_orientation_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_recognizer_GPU_options = gpuModuleOptions
options.GPU_options.face_template_quality_estimator_GPU_options = gpuModuleOptions 
options.GPU_options.mask_detector_GPU_options = gpuModuleOptions
options.GPU_options.object_detector_GPU_options = gpuModuleOptions
options.GPU_options.spoof_detector_GPU_options = gpuModuleOptions


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
    print(f"{Fore.RED}Unable to open video stream{Style.RESET_ALL}")
    quit()

# Get the original video resolution
res_w  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
res_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Original resolution: (", res_w, "x", res_h, ")")

# Try to use HD resolution, will use closes available resolution to this
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
res_w  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
res_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Set resolution to: (", res_w, "x", res_h, ")")

while(True):
    # To skip some frames, uncomment the following
    # cap.grab()

    ret, frame = cap.read()
    if ret == False:
        continue

    # Set the image using the frame buffer. OpenCV stores images in BGR format
    res, img = sdk.preprocess_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set frame{Style.RESET_ALL}")
        continue

    # Run face detection
    res, found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if res != tfsdk.ERRORCODE.NO_ERROR:
        print(f'{Fore.RED}Unable to detect face: {res.name}{Style.RESET_ALL}')
        continue

    if not found:
        # No face was found in the frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Run blink detection
    res, blinkstate = sdk.detect_blink(img, face_box_and_landmarks)

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




