# Sample code: Demonstrates how to run active spoof.
# Active spoof works by analyzing the way a persons face changes as they move closer to a camera.
# The active spoof solution therefore required two images and expects the face a certain distance from the camera.
# In the far image, the face should be about 18 inches from the camera, while in the near image,
# the face should be 7-8 inches from the camera.

# In this sample app, we run spoof detection using both a real image pair and spoof attempt image pair.

# You can find a more involved sample app here: https://github.com/getchui/sample-apps/tree/master/python/active_spoof_frontend_app

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
options.initialize_module.active_spoof = True

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
options.GPU_options.object_detector_GPU_options = gpuModuleOptions

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

# Start by analyzing the real images
# We will start with the far shot image
res, img = sdk.preprocess_image("../images/far_shot_real_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set real image, far shot{Style.RESET_ALL}")
    quit()


# Next, must detect if there is a face in the image
found, fb = sdk.detect_largest_face(img)
if found == False:
    print(f"{Fore.RED}Unable to find face in real image, far shot{Style.RESET_ALL}")
    quit()

# Now, we must ensure that the face meets the size criteria
# Be sure to check the return value of this function

ret = sdk.check_spoof_image_face_size(img, fb, tfsdk.ACTIVESPOOFSTAGE.FAR)
if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
    print(f"{Fore.RED}The face is too far in the real image, far face{Style.RESET_ALL}")
    quit()
elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
    print(f"{Fore.RED}The face is too close in the real image, far face{Style.RESET_ALL}")
    quit()
elif ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error with the real image, far face{Style.RESET_ALL}")
    quit()

# Next, we need to obtain the 106 facial landmarks for the face. 
ret, far_landmarks = sdk.get_face_landmarks(img,fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the real image, far face{Style.RESET_ALL}")
    quit()    

# Finally, we can compute a face recognition template for the face, 
# and later use it to ensure the two active spoof images are from the same identity.
ret, far_faceprint = sdk.get_face_feature_vector(img, fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the real image, far face{Style.RESET_ALL}")
    quit()


# Now at this point, we can repeat all the above steps, but for the near image now. 
res, img = sdk.preprocess_image("../images/near_shot_real_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set real image, near shot{Style.RESET_ALL}")
    quit()

found, fb = sdk.detect_largest_face(img)
if found == False:
    print(f"{Fore.RED}Unable to find face in real image, near shot{Style.RESET_ALL}")
    quit()

ret = sdk.check_spoof_image_face_size(img, fb, tfsdk.ACTIVESPOOFSTAGE.NEAR) # Be sure to specify near image this time.
if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
    print(f"{Fore.RED}The face is too far in the real image, near face{Style.RESET_ALL}")
    quit()
elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
    print(f"{Fore.RED}The face is too close in the real image, near face{Style.RESET_ALL}")
    quit()
elif ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error with the real image, near face{Style.RESET_ALL}")
    quit()

ret, near_landmarks = sdk.get_face_landmarks(img, fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the real image, near face{Style.RESET_ALL}")
    quit()    

ret, near_faceprint = sdk.get_face_feature_vector(img, fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the real image, near face{Style.RESET_ALL}")
    quit()

# At this point, we can finally run active spoof detection
ret, spoof_score, spoof_label = sdk.detect_active_spoof(near_landmarks, far_landmarks)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run active spoof detection for real images!{Style.RESET_ALL}")
    quit()

print("Printing results for real image:")

if spoof_label == tfsdk.SPOOFLABEL.FAKE:
    print("SPOOF RESULTS: Spoof attempt detected!")
else:
    # Finally, as a last step, we can compare the two face recognition templates to ensure they are of the same identity
    ret, match_prob, sim_score = sdk.get_similarity(near_faceprint, far_faceprint)
    if sim_score < 0.3:
        print("SPOOF RESULTS: Real image detected, but images are of different identities!")
    else:
        print("SPOOF RESULTS: Real image detected, and both images are of the same identity!")

print("")

# Now for the sake of the demo, let's repeat the entire process but with two face / spoof attempt images
# We will start with the far shot image
res, img = sdk.preprocess_image("../images/far_shot_fake_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set fake image, far shot{Style.RESET_ALL}")
    quit()


# Next, must detect if there is a face in the image
found, fb = sdk.detect_largest_face(img)
if found == False:
    print(f"{Fore.RED}Unable to find face in fake image, far shot{Style.RESET_ALL}")
    quit()

# Now, we must ensure that the face meets the size criteria
# Be sure to check the return value of this function

ret = sdk.check_spoof_image_face_size(img, fb, tfsdk.ACTIVESPOOFSTAGE.FAR)
if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
    print(f"{Fore.RED}The face is too far in the fake image, far face{Style.RESET_ALL}")
    quit()
elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
    print(f"{Fore.RED}The face is too far in the fake image, far face{Style.RESET_ALL}")
    quit()
elif ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error with the fake image, far face{Style.RESET_ALL}")
    quit()

# Next, we need to obtain the 106 facial landmarks for the face. 
ret, far_landmarks = sdk.get_face_landmarks(img, fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the fake image, far face{Style.RESET_ALL}")
    quit()    

# Finally, we can compute a face recognition template for the face, 
# and later use it to ensure the two active spoof images are from the same identity.
ret, far_faceprint = sdk.get_face_feature_vector(img, fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the fake image, far face{Style.RESET_ALL}")
    quit()


# Now at this point, we can repeat all the above steps, but for the near image now. 
res, img = sdk.preprocess_image("../images/near_shot_fake_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set real image, near shot{Style.RESET_ALL}")
    quit()

found, fb = sdk.detect_largest_face(img)
if found == False:
    print(f"{Fore.RED}Unable to find face in fake image, near shot{Style.RESET_ALL}")
    quit()

ret = sdk.check_spoof_image_face_size(img, fb, tfsdk.ACTIVESPOOFSTAGE.NEAR) # Be sure to specify near image this time.
if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
    print(f"{Fore.RED}The face is too far in the fake image, near face{Style.RESET_ALL}")
    quit()
elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
    print(f"{Fore.RED}The face is too far in the fake image, near face{Style.RESET_ALL}")
    quit()
elif ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error with the fake image, near face{Style.RESET_ALL}")
    quit()

ret, near_landmarks = sdk.get_face_landmarks(img, fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the fake image, near face{Style.RESET_ALL}")
    quit()    

ret, near_faceprint = sdk.get_face_feature_vector(img,fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the fake image, near face{Style.RESET_ALL}")
    quit()

# At this point, we can finally run active spoof detection
ret, spoof_score, spoof_label = sdk.detect_active_spoof(near_landmarks, far_landmarks)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run active spoof detection for fake images!{Style.RESET_ALL}")
    quit()

print("Printing results for fake image:")

if spoof_label == tfsdk.SPOOFLABEL.FAKE:
    print("SPOOF RESULTS: Spoof attempt detected!")
else:
    # Finally, as a last step, we can compare the two face recognition templates to ensure they are of the same identity
    ret, match_prob, sim_score = sdk.get_similarity(near_faceprint, far_faceprint)
    if sim_score < 0.3:
        print("SPOOF RESULTS: Real image detected, but images are of different identities!")
    else:
        print("SPOOF RESULTS: Real image detected, and both images are of the same identity!")