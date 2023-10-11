# Sample code: Generate face recognition templates using GPU batching.

# This sample app demonstrates how to use batching with the GPU SDK.
# First, we extract the face chip for several images
# Next, we generate face recognition templates in batch.
# Batching increases the GPU throughput.
# Finally, we generate the similarity scores.

import tfsdk
import os
from colorama import Fore
from colorama import Style

# In this sample app, will use a batch size of 3
batch_size = 3

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
options.GPU_options = True;
options.GPU_options.device_index = 0

gpuModuleOptions = tfsdk.GPUModuleOptions()
gpuModuleOptions.max_batch_size = batch_size
gpuModuleOptions.opt_batch_size = batch_size
gpuModuleOptions.max_workspace_size = 2000
gpuModuleOptions.precision = tfsdk.PRECISION.FP16

# Note, you can set separate GPU options for each GPU supported module
options.GPU_options.face_detector_GPU_options = gpuModuleOptions
options.GPU_options.face_recognizer_GPU_options = gpuModuleOptions
options.GPU_options.mask_detector_GPU_options = gpuModuleOptions
options.GPU_options.object_detector_GPU_options = gpuModuleOptions

# You can also enable GPU for all supported modules at once through the following syntax
# options.GPU_options = True

sdk = tfsdk.SDK(options)
# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()



# List of images to use for our batch template generation
images = [
    "../images/brad_pitt_1.jpg",
    "../images/brad_pitt_2.jpg",
    "../images/brad_pitt_3.jpg"
]

# Create array to store our face chips
face_chips = []

# First need to run face detection and generate face chips sequentially.
for image in images:
    res, img = sdk.preprocess_image(image)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set image: {image}{Style.RESET_ALL}")
        quit()


     # Detect the largest face in the image
    found, face_bounding_box = sdk.detect_largest_face(img)

    if not found:
        print(f"{Fore.RED}Could not find face in image!{Style.RESET_ALL}")
        quit()


    face_chip = sdk.extract_aligned_face(img, face_bounding_box)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to extract aligned face{Style.RESET_ALL}")
        quit()

    face_chips.append(face_chip)

# Run mask detection in batch
res, mask_labels, mask_scores = sdk.detect_masks(face_chips)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to run mask detection{Style.RESET_ALL}")
    quit()

for mask_label, mask_score in zip(mask_labels, mask_scores):
    if mask_label == tfsdk.MASKLABEL.MASK:
        print(f"Masked image detected with probability of {1.0-mask_score:0.3f}")
    else:
        print(f"Unmasked image detected with probability of {mask_score:0.3f}")

# Now that we have generated the face chips, we can go ahead and batch generate the FR templates.
res, faceprints = sdk.get_face_feature_vectors(face_chips)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to extract face feature vectors{Style.RESET_ALL}")
    quit()

# Run similarity comparisons
res, match_prob, sim_score = sdk.get_similarity(faceprints[0], faceprints[1])
print(f"Image 1 vs Image 2 match probability: {match_prob}")

res, match_prob, sim_score = sdk.get_similarity(faceprints[1], faceprints[2])
print(f"Image 2 vs Image 3 match probability: {match_prob}")

