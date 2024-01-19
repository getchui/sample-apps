# Sample code: Load collection from disk then run 1N identification using frames from the webcame videostream

# This sample app demonstrates how to use 1N identification. First, an existing collection (created by running enroll_in_database) is loaded from disk.
# Next, 1N identification is run to determine the identity of an anonymous template.
# Note, you must run enroll_in_database before being able to run this sample app.

# Note: you will need to have the opencv-python module installed

import cv2
import os
import sys
import tfsdk
import time
from colorama import Fore
from colorama import Style

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
options.smallest_face_height = 40
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
options.initialize_module.face_detector = True
options.initialize_module.face_recognizer = True

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


# Connect to our existing database
res = sdk.create_database_connection("my_database.db")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
    quit()



# ex. If using POSTGRESQL backend...
# res = sdk.create_database_connection("host=localhost port=5432 dbname=my_database user=postgres password=admin")
# if (res != tfsdk.ERRORCODE.NO_ERROR):
  # print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
  # quit()


# Load the existing collection into memory
res = sdk.create_load_collection("my_collection")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to load collection{Style.RESET_ALL}")
    quit()


filename = "my_window"

# Use the default camera (TODO: Can change the camera source, for example to an RTSP stream)
cap = cv2.VideoCapture(0)
if (cap.isOpened()== False):
    print(f"{Fore.RED}Error opening video stream{Style.RESET_ALL}")
    os._exit(1)


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
        print(f"{Fore.RED}Unable to set frame.{Style.RESET_ALL}")
        cv2.imshow(filename, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    res, faceboxes = sdk.detect_faces(img)
    if res != tfsdk.ERRORCODE.NO_ERROR:
        print(f'{Fore.RED}Unable to detect faces: {res.name}{Style.RESET_ALL}')
        cv2.imshow(filename, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if not faceboxes or len(faceboxes) == 0:
        cv2.imshow(filename, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    annotation_data = []

    # Run 1:N search for all extracted faces
    for facebox in faceboxes:
        # Extract the feature vector
        # We do not need to check the quality of the probe face templates
        # We mainly want to ensure that the enrollment templates are high quality,
        # This is less of a concern with probe templates
        res, faceprint = sdk.get_face_feature_vector(img, facebox)
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("skipping facebox")
            continue

        res, match_found, candidate = sdk.identify_top_candidate(faceprint, threshold=0.4)
        if (res == tfsdk.ERRORCODE.NO_ERROR):
            annotation_data.append((facebox, match_found, candidate))

    for facebox, match_found, candidate in annotation_data:
        if match_found:
            # Draw a rectangle with the ID and match score
            res = sdk.draw_candidate_bounding_box_and_label(img, facebox, candidate)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print(f"{Fore.RED}Unable to draw candidate label and bounding box{Style.RESET_ALL}")
        else:
            # Draw a red rectangle around the face
            res = sdk.draw_face_box_and_landmarks(img, facebox, False, tfsdk.ColorRGB(255, 0, 0), 2)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print(f"{Fore.RED}Unable to draw bounding box{Style.RESET_ALL}")


    # Convert the annotated frame back to a OpenCV frame
    frame = img.as_numpy_array()


    cv2.imshow(filename, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




