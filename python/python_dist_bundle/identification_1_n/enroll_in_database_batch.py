# Sample code: Batch enroll face recognition templates for images and then enroll them into a collection.

# This sample app demonstrates how you can batch enroll a folder of identities' face recognition templates or Faceprints into a collection on disk.
# First, we create a database and create a new collection within that database.
# Next, we generate face recognition templates and enroll those templates into the collection.
# Note, after running this sample app, you can run the identification_1_n sample app.
# Folder structure format:
# root_folder
# |
# └───person_one_folder
# |   |
# |   └───photo1.jpg
# |       photo2.jpg
# |
# └───person_two_folder
# |   |
# |   └───photo1.jpg
# |       photo2.jpg
# |
# └───person_three_folder
# |   |
# |   └───photo1.jpg
# |       photo2.jpg
# |       photo3.jpg
# |
# The subfolder names are used as the enrollment identity string for the photos in each subfolder
# Command line format:
# python enroll_in_database_batch.py root_folder 

import tfsdk
import os
from imutils import paths
import math
import sys
import subprocess
from colorama import Fore
from colorama import Style

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
options.smallest_face_height = -1 
# The path specifying the directory containing the model files which were downloaded.
options.models_path = "./"
# Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
options.fr_vector_compression = False
# Database management system for the storage of biometric templates for 1 to N identification.
options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE
# If you want to use Postgresql database options, comment the above line and comment out below
# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.POSTGRESQL
is_db_sql = False if options.dbms == tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE else True
# Please change the following variable to your database parameters
SDK_DB_NAME = "SDK_DB_NAME"
SDK_DB_HOST = "LOCALHOST"
SDK_DB_PORT = 1234
SDK_DB_PASS = "USERNAME"
SDK_DB_USER = "PASSWORD"
db_connection_string = f"host={SDK_DB_HOST} port={SDK_DB_PORT} user={SDK_DB_USER} password={SDK_DB_PASS} dbname={SDK_DB_NAME}"

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

# Create a new database
res = sdk.create_database_connection("my_database.db" if not is_db_sql else db_connection_string)
if (res != tfsdk.ERRORCODE.NO_ERROR):
  print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
  quit()

# ex. If using POSTGRESQL backend...
# res = sdk.create_database_connection("host=localhost port=5432 dbname=my_database user=postgres password=admin")
# if (res != tfsdk.ERRORCODE.NO_ERROR):
#   print("Unable to create database connection")
#   quit()

# Create a new collection
res = sdk.create_load_collection("my_collection")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to create collection{Style.RESET_ALL}")
    quit()

# Since our collection is empty, lets populate the collection with some identities
folder_location = sys.argv[1]
if os.path.isdir(folder_location):
  images = sorted(list(paths.list_images(folder_location)))
  labels = [os.path.basename(os.path.dirname(image)) for image in images]
  image_identities = list(zip(images, labels))
else:
  print(f"{Fore.RED}Unable to verify folder{Style.RESET_ALL}")  
  quit() 

failed_enrollment = []

for path, identity in image_identities:
    print("Processing image:", path, "with identity:", identity)
    # Generate a template for each image
    res, img = sdk.preprocess_image(path)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        failed_enrollment.append(path)
        print(f"{Fore.RED}Unable to set image at path: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # Detect the largest face in the image
    found, faceBoxAndLandmarks = sdk.detect_largest_face(img)
    if found == False:
        failed_enrollment.append(path)
        print(f"{Fore.RED}No face detected in image: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # We want to only enroll high quality images into the database / collection
    # For more information, refer to the section titled "Selecting the Best Enrollment Images"
    # https://reference.trueface.ai/cpp/dev/latest/py/identification.html

    # Therefore, ensure that the face height is at least 100px
    faceHeight = faceBoxAndLandmarks.bottom_right.y - faceBoxAndLandmarks.top_left.y
    print(f"Face height: {faceHeight} pixels")

    if faceHeight < 100:
        failed_enrollment.append(path)
        print(f"{Fore.RED}The face is too small in the image for a high quality enrollment, not enrolling{Style.RESET_ALL}")
        continue

    # Get the aligned chip so we can compute the image quality
    face = sdk.extract_aligned_face(img, faceBoxAndLandmarks)

    # We can check the orientation of the head and ensure that it is facing forward
    # To see the effect of yaw and pitch on match score, refer to: https://reference.trueface.ai/cpp/dev/latest/py/face.html#tfsdk.SDK.estimate_head_orientation

    res, yaw, pitch, roll = sdk.estimate_head_orientation(img, faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        failed_enrollment.append(path)
        print(f"{Fore.RED}Unable to compute head orientation, not enrolling{Style.RESET_ALL}")
        continue

    yaw_deg = yaw * 180 / 3.14
    pitch_deg = pitch * 180 / 3.14

    if abs(yaw_deg) > 50:
        failed_enrollment.append(path)
        print(f"{Fore.RED}Enrollment image has too extreme a yaw, not enrolling{Style.RESET_ALL}")
        continue

    if abs(pitch_deg) > 35:
        failed_enrollment.append(path)
        print(f"{Fore.RED}Enrollment image has too extreme a pitch, not enrolling{Style.RESET_ALL}")
        continue

    # Finally ensure the user is not wearing a mask
    error_code, mask_label = sdk.detect_mask(img, faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        failed_enrollment.append(path)
        print(f"{Fore.RED}Unable to run mask detection, not enrolling{Style.RESET_ALL}")
        continue

    if (mask_label == tfsdk.MASKLABEL.MASK):
        failed_enrollment.append(path)
        print(f"{Fore.RED}Please choose a image without a mask for enrollment, not enrolling{Style.RESET_ALL}")
        continue

    # Now that we have confirmed the images are high quality, generate a template from that image
    res, faceprint = sdk.get_face_feature_vector(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        failed_enrollment.append(path)
        print(f"{Fore.RED}There was an error generating the faceprint, not enrolling{Style.RESET_ALL}")
        continue

    # Enroll the feature vector into the collection
    res, UUID = sdk.enroll_faceprint(faceprint, identity)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        failed_enrollment.append(path)
        print(f"{Fore.RED}Unable to enroll feature vector{Style.RESET_ALL}")
        continue

    # TODO: Can store the UUID for later use
    print(f"Success, enrolled template with UUID: {UUID}")
    print("--------------------------------------------")
    print()

# Print all the images that failed enrollment
print(f"{Fore.RED}Unable to enroll the following images{Style.RESET_ALL}")
for failed_img in failed_enrollment:
  print(failed_img)

# For the sake of the demo, print the information about the collection
res, collection_names = sdk.get_collection_names()
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to get collection names{Style.RESET_ALL}")
    quit()

for collection_name in collection_names:
    res, metadata = sdk.get_collection_metadata(collection_name)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to get collection metadata{Style.RESET_ALL}")
        quit()

    print(f"Metadata: {metadata}")

    res, identities = sdk.get_collection_identities(collection_name)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to get collection identities{Style.RESET_ALL}")
        quit()

    print(f"Identities: {identities}")
