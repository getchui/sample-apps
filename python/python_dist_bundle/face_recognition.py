# Sample code: Load two images from disk, extract feature vectors and compare similarity

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
options.models_path = "./"
# Enable vector compression to improve 1 to 1 comparison speed and 1 to N search speed.
options.fr_vector_compression = False
# Database management system for the storage of biometric templates for 1 to N identification.
options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE

# Encrypt the biometric templates stored in the database
# If encryption is enabled, must provide an encryption key
options.encryptDatabase.enable_encryption = False
options.encryptDatabase.key = "TODO: Your encryption key here"

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
gpuOptions = tfsdk.GPUOptions()
gpuOptions.enable_GPU = False # TODO: Change this to true to enable GPU
gpuOptions.max_batch_size = 4
gpuOptions.opt_batch_size = 1
gpuOptions.max_workspace_size = 2000
gpuOptions.device_index = 0
gpuOptions.precsion = tfsdk.PRECISION.FP16

# Note, you can set separate GPU options for each GPU supported module
options.GPU_options.face_detector_GPU_options = gpuOptions
options.GPU_options.face_recognizer_GPU_options = gpuOptions

# You can also enable GPU for all supported modules at once through the following syntax
# options.GPU_options = True

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

# Set the first image
res = sdk.set_image("../images/brad_pitt_1.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()


# Extract the feature vector
res, v1, found = sdk.get_largest_face_feature_vector()
if (res != tfsdk.ERRORCODE.NO_ERROR or found == False):
    print(f"{Fore.RED}Unable to generate feature vector 1, no face detected{Style.RESET_ALL}")
    quit()

# Set the second image
res = sdk.set_image("../images/brad_pitt_2.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 2{Style.RESET_ALL}")
    quit()

res, v2, found = sdk.get_largest_face_feature_vector()
if (res != tfsdk.ERRORCODE.NO_ERROR or found == False):
    print(f"{Fore.RED}Unable to generate feature vector 2, no face detected{Style.RESET_ALL}")
    quit()

res, match_probability, similarity = sdk.get_similarity(v1, v2)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to compute similarity{Style.RESET_ALL}")
    quit()

print(f"Match Probability: {match_probability}")
print(f"Simlarity Score: {similarity}")


