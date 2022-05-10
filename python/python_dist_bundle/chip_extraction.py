# Sample code: Extract and display the aligned 112x112 pixel face image

import tfsdk
import os
from colorama import Fore
from colorama import Style
import cv2

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

# Load the input image
res, img = sdk.preprocess_image("../images/brad_pitt_1.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# detect the largest face
found, face_bounding_box = sdk.detect_largest_face(img)


if found:
    face = sdk.extract_aligned_face(img, face_bounding_box)
    
    # Save the chip to disk
    face.save_image("facechip.jpg")
else:
    print("No face detected in image")


# Will demonstrate how to manually set the landmark locations if you are using a dataset which provides them
# Doing so can bypass the face detection step and speed things up
fb = tfsdk.FaceBoxAndLandmarks()

fb.top_left.x = 176.822021
fb.top_left.y = 160.230774
fb.bottom_right.x = 413.943542
fb.bottom_right.y = 483.534943

fb.score = 0.992754

p1 = tfsdk.Point()
p2 = tfsdk.Point()
p3 = tfsdk.Point()
p4 = tfsdk.Point()
p5 = tfsdk.Point()

p1.x = 224.090729
p1.y = 286.082031

p2.x = 331.095398
p2.y = 284.688293

p3.x = 261.967041
p3.y = 338.466797

p4.x = 230.183548
p4.y = 399.291016

p5.x = 318.272552
p5.y = 397.987305

fb.landmarks = [p1, p2, p3, p4, p5]

face = sdk.extract_aligned_face(img, fb)
