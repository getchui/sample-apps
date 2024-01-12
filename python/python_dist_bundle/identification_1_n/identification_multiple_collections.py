# Sample code: Generate face recognition templates for images and then enroll them into two different collections, and then query each collection.

import tfsdk
import os
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

# Create a new database
res = sdk.create_database_connection("multiple_collections.db")
if (res != tfsdk.ERRORCODE.NO_ERROR):
  print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
  quit()

# ex. If using POSTGRESQL backend...
# res = sdk.create_database_connection("host=localhost port=5432 dbname=my_database user=postgres password=admin")
# if (res != tfsdk.ERRORCODE.NO_ERROR):
#   print("Unable to create database connection")
#   quit()

# Create the two collections 
collection_1 = "collection_1"
collection_2 = "collection_2"

res = sdk.create_collection(collection_1)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to create collection{Style.RESET_ALL}")
    quit()

res = sdk.create_collection(collection_2)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to create collection{Style.RESET_ALL}")
    quit()


# Load the two collections into memory
res = sdk.load_collections([collection_1, collection_2])
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to load collections into memory{Style.RESET_ALL}")
    quit()


# Enroll Faceprint of Brad Pitt in collection 1
res, img = sdk.preprocess_image("../images/brad_pitt_1.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image{Style.RESET_ALL}")
    quit()

res, fp, found = sdk.get_largest_face_feature_vector(img)
if (res != tfsdk.ERRORCODE.NO_ERROR or not found):
    print(f"{Fore.RED}Unable to find face in image{Style.RESET_ALL}")
    quit()

# Since we have muliple collections loaded in memory, we must specify the collection name for enrollment
res, UUID = sdk.enroll_faceprint(fp, "Brad Pitt", collection_1)

# Enroll Faceprint of Tom Cruise in collection 2
res, img = sdk.preprocess_image("../images/tom_cruise_1.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image{Style.RESET_ALL}")
    quit()

res, fp, found = sdk.get_largest_face_feature_vector(img)
if (res != tfsdk.ERRORCODE.NO_ERROR or not found):
    print(f"{Fore.RED}Unable to find face in image{Style.RESET_ALL}")
    quit()

# Since we have muliple collections loaded in memory, we must specify the collection name for enrollment
res, UUID = sdk.enroll_faceprint(fp, "Tom Cruise", collection_2)

# Now run a search query in collection 1
res, img = sdk.preprocess_image("../images/brad_pitt_2.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image{Style.RESET_ALL}")
    quit()

res, fp, found = sdk.get_largest_face_feature_vector(img)
if (res != tfsdk.ERRORCODE.NO_ERROR or not found):
    print(f"{Fore.RED}Unable to find face in image{Style.RESET_ALL}")
    quit()

# Since we have muliple collections loaded in memory, we must specify the collection name for identification
res, found, candidate = sdk.identify_top_candidate(fp, 0.4, collection_1)
if res != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run identification query{Style.RESET_ALL}")
    quit()

if found:
    print("Found match candidate in collection:", collection_1, "with identity:", candidate.identity, "and match probability:", candidate.match_probability)

else:
    print("Unable to find match candidate in collection:", collection_1)

# Now run a search query in collection 2
res, img = sdk.preprocess_image("../images/tom_cruise_2.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image{Style.RESET_ALL}")
    quit()

res, fp, found = sdk.get_largest_face_feature_vector(img)
if (res != tfsdk.ERRORCODE.NO_ERROR or not found):
    print(f"{Fore.RED}Unable to find face in image{Style.RESET_ALL}")
    quit()

# Since we have muliple collections loaded in memory, we must specify the collection name for identification
res, found, candidate = sdk.identify_top_candidate(fp, 0.4, collection_2)
if res != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run identification query{Style.RESET_ALL}")
    quit()

if found:
    print("Found match candidate in collection:", collection_2, "with identity:", candidate.identity, "and match probability:", candidate.match_probability)

else:
    print("Unable to find match candidate in collection:", collection_2)