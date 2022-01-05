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

def draw_label(image, point, label,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.0, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x_label, y_label = point
    cv2.rectangle(
        image,
        (x_label, y_label - size[1] - 10),
        (x_label + size[0], y_label),
        (194,134,58),
        cv2.FILLED)

    cv2.putText(
        image, label.capitalize(), (x_label, y_label - 5), font, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA)


def draw_rectangle(frame, bounding_box):

    # Draw the rectangle on the frame
    cv2.rectangle(frame,
                  (int(bounding_box.top_left.x), int(bounding_box.top_left.y)),
                  (int(bounding_box.bottom_right.x), int(bounding_box.bottom_right.y)), (194,134,58), 3)
        


options = tfsdk.ConfigurationOptions()
# Can set configuration options here

# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.NONE # The data will not persist after the application terminates

# Load the collection from an SQLITE database
options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE

# If you previously enrolled the templates into a PostgreSQL database, then use POSTGRESQL instead
# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.POSTGRESQL

options.smallest_face_height = 40 # https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface20ConfigurationOptions18smallestFaceHeightE
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.GPU_options = True

# To enable database encryption...
# encryptDatabase = tfsdk.EncryptDatabase()
# encryptDatabase.enable_encryption = True
# encryptDatabase.key = "TODO: Your encryption key here"
# options.encrypt_database = encryptDatabase

# Since we know we will use the face detector and face recognizer,
# we can choose to initialize these modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.face_detector = True
initializeModule.face_recognizer = True
options.initialize_module = initializeModule

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
        cv2.imshow(filename, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    faceboxes = sdk.detect_faces()
    if not faceboxes or len(faceboxes) == 0:
        cv2.imshow(filename, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Run 1:N search for all extracted faces
    for facebox in faceboxes:
        # Extract the feature vector
        # We do not need to check the quality of the probe face templates
        # We mainly want to ensure that the enrollment templates are high quality, 
        # This is less of a concern with probe templates 
        res, faceprint = sdk.get_face_feature_vector(facebox)
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("skipping facebox")
            continue

        res, match_found, candidate = sdk.identify_top_candidate(faceprint, threshold=0.4)
        draw_rectangle(frame, facebox)

        if (res != tfsdk.ERRORCODE.NO_ERROR or not match_found):
            continue
        else:
            draw_label(frame,
                       (int(facebox.top_left.x), int(facebox.top_left.y)),
                       "{} {}%".format(
                           candidate.identity,
                           int(candidate.match_probability*100)))


    cv2.imshow(filename, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




