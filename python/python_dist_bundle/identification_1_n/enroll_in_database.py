# Sample code: Generate face recognition templates for images and then enroll them into a collection.

# This sample app demonstrates how you can enroll face recognition templates or Faceprints into a collection on disk.
# First, we create a database and create a new collection within that database.
# Next, we generate face recognition templates and enroll those templates into the collection.
# Note, after running this sample app, you can run the identification_1_n sample app.
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
options.smallest_face_height = 80 # Set this to 80 because we only want to enroll high quality images
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
options.initialize_module.face_blur_detector = True
options.initialize_module.face_template_quality_estimator = True
options.initialize_module.face_orientation_detector = True

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
res = sdk.create_database_connection("my_database.db")
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
image_identities = [
    ("../images/brad_pitt_1.jpg", "Brad Pitt"),
    ("../images/brad_pitt_2.jpg", "Brad Pitt"), # Can add the same identity more than once
    ("../images/tom_cruise_1.jpg", "Tom Cruise")
]

for path, identity in image_identities:
    print("Processing image:", path, "with identity:", identity)
    # Generate a template for each image
    res, img = sdk.preprocess_image(path)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set image at path: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # Since we are enrolling images from disk, there is a possibility that the images may be oriented incorrectly.
    # Therefore, run the orientation detector and adjust for any needed rotation
    ret, rotation = sdk.get_face_image_rotation(img)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}There was an error computing the image rotation{Style.RESET_ALL}")
        continue

    # Adjust for rotation
    img.rotate(rotation)

    # Detect the largest face in the image
    res, found, faceBoxAndLandmarks = sdk.detect_largest_face(img)
    if res != tfsdk.ERRORCODE.NO_ERROR or found == False:
        print(f"{Fore.RED}No face detected in image: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # We want to only enroll high quality images into the database / collection
    # For more information, refer to the section titled "Selecting the Best Enrollment Images"
    # https://reference.trueface.ai/cpp/dev/latest/py/identification.html

    # Ensure that the image is not too bright or dark, and that the exposure is optimal for face recognition
    res, quality, percentImageBright, percentImageDark, percentFaceBright = sdk.check_face_image_exposure(img, faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to get the face image exposure, not enrolling{Style.RESET_ALL}")
        continue

    if (quality != tfsdk.FACEIMAGEQUALITY.GOOD):
        print(f"{Fore.RED}The face image is over or under exposed, not enrolling{Style.RESET_ALL}")
        continue


    # Get the aligned chip so we can compute the image blur
    res, face = sdk.extract_aligned_face(img, faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to extract aligned face: {res.name}, not entrolling{Style.RESET_ALL}")
        continue

    # Ensure the face chip will generate a good face recognition template
    res, is_good_quality, score = sdk.estimate_face_template_quality(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}There was an error estimating the face template quality, not enrolling{Style.RESET_ALL}")
        continue

    if (not is_good_quality):
        print(f"{Fore.RED}The face chip is not suitable for face recognition template generation, not enrolling{Style.RESET_ALL}")
        continue

    # Ensure the face image is not too blurry
    res, quality, score = sdk.detect_face_image_blur(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}There was an error computing the image blur, not enrolling{Style.RESET_ALL}")
        continue

    if (quality != tfsdk.FACEIMAGEQUALITY.GOOD):
        print(f"{Fore.RED}The face image is too blurry, not enrolling{Style.RESET_ALL}")
        continue

    # We can check the orientation of the head and ensure that it is facing forward
    # To see the effect of yaw and pitch on match score, refer to: https://reference.trueface.ai/cpp/dev/latest/py/face.html#tfsdk.SDK.estimate_head_orientation
    ret, landmarks = sdk.get_face_landmarks(img, faceBoxAndLandmarks)
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to get face landmarks, not enrolling")
        continue

    res, yaw, pitch, roll, rot_vec, trans_vec = sdk.estimate_head_orientation(img, faceBoxAndLandmarks, landmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to compute head orientation, not enrolling{Style.RESET_ALL}")
        continue

    yaw_deg = yaw * 180 / 3.14
    pitch_deg = pitch * 180 / 3.14

    if abs(yaw_deg) > 30:
        print(f"{Fore.RED}Enrollment image has too extreme a yaw, not enrolling{Style.RESET_ALL}")
        continue

    if abs(pitch_deg) > 30:
        print(f"{Fore.RED}Enrollment image has too extreme a pitch, not enrolling{Style.RESET_ALL}")
        continue

    # Finally ensure the user is not wearing a mask
    error_code, mask_label, mask_score = sdk.detect_mask(img, faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run mask detection, not enrolling{Style.RESET_ALL}")
        continue

    if (mask_label == tfsdk.MASKLABEL.MASK):
        print(f"{Fore.RED}Please choose a image without a mask for enrollment, not enrolling{Style.RESET_ALL}")
        continue

    # Now that we have confirmed the images are high quality, generate a template from that image
    res, faceprint = sdk.get_face_feature_vector(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}There was an error generating the faceprint, not enrolling{Style.RESET_ALL}")
        continue

    # Enroll the feature vector into the collection
    res, UUID = sdk.enroll_faceprint(faceprint, identity)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to enroll feature vector{Style.RESET_ALL}")
        continue

    # TODO: Can store the UUID for later use
    print(f"Success, enrolled template with UUID: {UUID}")
    print("--------------------------------------------")
    print()


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
