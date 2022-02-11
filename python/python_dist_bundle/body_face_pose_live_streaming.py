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


# Start by specifying the configuration options to be used. 
# Can choose to use the default configuration options if preferred by calling the default SDK constructor.
# Learn more about the configuration options: https://reference.trueface.ai/cpp/dev/latest/py/general.html
options = tfsdk.ConfigurationOptions()
# The face recognition model to use. Use the most accurate model TFV5.
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5
# The object detection model to use.
options.obj_model = tfsdk.OBJECTDETECTIONMODEL.ACCURATE
# The face detection filter.
options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
# Smallest face height in pixels for the face detector.
# Can set this to -1 to dynamically change the smallest face height based on the input image size.
options.smallest_face_height = 40 
# The path specifying the directory containing the model files which were downloaded.
options.models_path = "./"
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
options.initialize_module.face_detector = True
options.initialize_module.bodypose_estimator = True

# Options for enabling GPU
# We will disable GPU inference, but you can easily enable it by modifying the following options
# Note, you may require a specific GPU enabled token in order to enable GPU inference.
gpuModuleOptions = tfsdk.GPUModuleOptions()
gpuModuleOptions.enable_GPU = False # TODO: Change this to true to enable GPU
gpuModuleOptions.max_batch_size = 4
gpuModuleOptions.opt_batch_size = 1
gpuModuleOptions.max_workspace_size = 2000
gpuModuleOptions.device_index = 0
gpuModuleOptions.precision = tfsdk.PRECISION.FP16

# Note, you can set separate GPU options for each GPU supported module
options.GPU_options.face_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_recognizer_GPU_options = gpuModuleOptions

# You can also enable GPU for all supported modules at once through the following syntax
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
    res, img = sdk.preprocess_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set frame.{Style.RESET_ALL}")
        continue

    # Detect the largest face, then compute the face pose
    found, faceBoxAndLandmarks = sdk.detect_largest_face(img)
    if found == True:
        res, yaw, pitch, roll = sdk.estimate_head_orientation(img,faceBoxAndLandmarks)
        if res == tfsdk.ERRORCODE.NO_ERROR:
            draw_pose_lines(yaw, pitch, roll, frame)

    # Run object detection, then run body pose estimation
    bounding_boxes = sdk.detect_objects(img)
    if (len(bounding_boxes) > 0) :
        body_landmarks = sdk.estimate_pose(img, bounding_boxes)
        if len(body_landmarks) > 0 :
            draw_body_pose(frame, body_landmarks)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




