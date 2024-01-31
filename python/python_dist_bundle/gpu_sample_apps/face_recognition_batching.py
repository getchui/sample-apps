# Sample code: Demonstrate batch inference.

import tfsdk
import os
from colorama import Fore
from colorama import Style

test_data = [
    "../images/brad_pitt_1.jpg",
    "../images/brad_pitt_2.jpg",
    "../images/brad_pitt_3.jpg",
    "../images/brad_pitt_4.jpg"
]

batch_size = len(test_data)

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
options.initialize_module.face_orientation_detector = True
options.initialize_module.landmark_detector = True
options.initialize_module.blink_detector = True
options.initialize_module.face_template_quality_estimator = True
options.initialize_module.face_blur_detector = True
options.initialize_module.mask_detector = True

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

# You can also enable GPU for all supported modules at once through the following syntax
# options.GPU_options = True

sdk = tfsdk.SDK(options)
# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()


tf_images = []

# Preprocess the images
for img_path in test_data:    
    ret, img = sdk.preprocess_image(img_path)
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print(f"{Fore.RED}Unable to set image: {img_path}{Style.RESET_ALL}")
        quit()

    tf_images.append(img)


# Run face image orientation detection 
ret, rotations = sdk.get_face_image_rotations(tf_images)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run face image rotation{Style.RESET_ALL}")
    quit()

# Adjust the images for any rotation issues
for i in range(len(rotations)):
    tf_images[i].rotate(rotations[i])

# Run face detection and extract the face chips
fbs = []
chips = []

for img in tf_images:
    ret, found, fb = sdk.detect_largest_face(img)
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print(f"{Fore.RED}Unable to run face detection{Style.RESET_ALL}")
        quit()

    if not found:
        print(f"{Fore.RED}Unable to find face in image, skipping{Style.RESET_ALL}")
        continue

    ret, chip = sdk.extract_aligned_face(img, fb)
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print(f"{Fore.RED}Unable extract face chip{Style.RESET_ALL}")
        quit()

    fbs.append(fb)
    chips.append(chip)

# Run 106 face landmark detection in batch
ret, landmarks_vec = sdk.get_face_landmarks(tf_images, fbs)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run face landmark detection{Style.RESET_ALL}")
    quit()

# Run blink detection in batch
ret, blink_states = sdk.detect_blinks(tf_images, landmarks_vec)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run blink detection{Style.RESET_ALL}")
    quit()


# Run mask detection in batch
ret, mask_labels, scores = sdk.detect_masks(chips)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run mask detection{Style.RESET_ALL}")
    quit()

# Run blur detection in batch
ret, face_qualitites, scores = sdk.detect_face_image_blurs(chips)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run face blur detection{Style.RESET_ALL}")
    quit()

# Run face template quality in batch
ret, are_template_qualities_good, scores = sdk.estimate_face_template_qualities(chips)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run face template quality detection{Style.RESET_ALL}")
    quit()


# Run face recognition in batch 
ret, faceprints = sdk.get_face_feature_vectors(chips)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to extract face feature vectors{Style.RESET_ALL}")
    quit()