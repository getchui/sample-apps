# Sample code: Load collection from disk then run 1N identification

# This sample app demonstrates how to use 1N identification. First, an existing collection (created by running enroll_in_database) is loaded from disk.
# Next, 1N identification is run to determine the identity of an anonymous template.
# Note, you must run enroll_in_database before being able to run this sample app. 

import tfsdk
import os
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
options.GPU_options.face_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_recognizer_GPU_options = gpuModuleOptions
options.GPU_options.mask_detector_GPU_options = gpuModuleOptions

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

# Use a new image of Brad Pitt as the probe
res, img = sdk.preprocess_image("../images/brad_pitt_1.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image{Style.RESET_ALL}")
    quit()

# Extract the feature vector
# We do not need to check the quality of the probe face templates
# We mainly want to ensure that the enrollment templates are high quality, 
# This is less of a concern with probe templates 
res, v1, found_face = sdk.get_largest_face_feature_vector(img)
if (res != tfsdk.ERRORCODE.NO_ERROR or found_face != True):
    print(f"{Fore.RED}No face found in image, unable to generate feature vector{Style.RESET_ALL}")
    quit()

# Run identification query using threshold of 0.4
res, found, candidate = sdk.identify_top_candidate(v1, 0.4)

if found == False or res != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to find match above threshold{Style.RESET_ALL}")
else:
    print("Match found!")
    print(f"Identity: {candidate.identity}")
    print(f"Match Probability: {candidate.match_probability}")
    print(f"Similarity: {candidate.similarity_measure}")