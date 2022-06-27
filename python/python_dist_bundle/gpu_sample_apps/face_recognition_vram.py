# Sample code: Load images into GPU memory then generate face recognition templates

# This sample app demonstrates how to use the SDK with images loaded in GPU memory. 
# First, cupy is used to load images into GPU memory.
# Next, the SDK is used to generate feature vectors for the two images. 


import os
import tfsdk
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
# Note, you may require a specific GPU enabled token in order to enable GPU inference.
options.GPU_options = True
options.GPU_options.device_index = 0

gpuModuleOptions = tfsdk.GPUModuleOptions()
gpuModuleOptions.max_batch_size = 4
gpuModuleOptions.opt_batch_size = 1
gpuModuleOptions.max_workspace_size = 2000
gpuModuleOptions.precision = tfsdk.PRECISION.FP16

# Note, you can set separate GPU options for each GPU supported module
options.GPU_options.face_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_recognizer_GPU_options = gpuModuleOptions
options.GPU_options.maks_detector_GPU_options = gpuModuleOptions

# You can also enable GPU for all supported modules at once through the following syntax
# options.GPU_options = True

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

import cv2
import numpy as np
img_1 = cv2.imread('../images/brad_pitt_1.jpg')
img_2 = cv2.imread('../images/brad_pitt_2.jpg')

import cupy as cp

# load the image into the graphics card's memory
img_gpu_1 = cp.asarray(img_1)

# pass the gpu memory address to the sdk
res, img = sdk.preprocess_image(img_gpu_1.data.ptr, img_1.shape[1], img_1.shape[0], tfsdk.COLORCODE.bgr, 0)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# generate the face feature vector from the GPU image
ret, fp1, found = sdk.get_largest_face_feature_vector(img)

if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the faceprint{Style.RESET_ALL}")
    quit()

if not found:
    print(f"{Fore.RED}Unable to find a face in image 1{Style.RESET_ALL}")
    quit()


# load the image into the graphics card's memory
img_gpu_2 = cp.asarray(img_2)

# pass the gpu memory address to the sdk
res, img = sdk.preprocess_image(img_gpu_2.data.ptr, img_2.shape[1], img_2.shape[0], tfsdk.COLORCODE.bgr, 0)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 2{Style.RESET_ALL}")
    quit()

# generate the face feature vector from the GPU image
ret, fp2, found = sdk.get_largest_face_feature_vector(img)

if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the faceprint{Style.RESET_ALL}")
    quit()

if not found:
    print(f"{Fore.RED}Unable to find a face in image 2{Style.RESET_ALL}")
    quit()

# compare the two faceprints, this should return a high match probability
ret, prob, sim = sdk.get_similarity(fp1, fp2)
print("Probability:", prob * 100, "%")
print("Similarity:", sim)
