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

options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.GPU_options = True

# Since we know we will use active spoof,
# we can choose to initialize this modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.active_spoof = True
options.initialize_module = initializeModule

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

# Start by analyzing the real images
# We will start with the far shot image
res = sdk.set_image("../images/far_shot_real_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set real image, far shot{Style.RESET_ALL}")
    quit()


# Next, we must obtain the image properties
# These properties are used by the check_spoof_image_face_size() function
image_props = sdk.get_image_properties()


# Next, must detect if there is a face in the image
found, fb = sdk.detect_largest_face()
if found == False:
    print(f"{Fore.RED}Unable to find face in real image, far shot{Style.RESET_ALL}")
    quit()

# Now, we must ensure that the face meets the size criteria
# Be sure to check the return value of this function

ret = sdk.check_spoof_image_face_size(fb, image_props, tfsdk.ACTIVESPOOFSTAGE.FAR)
if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
    print(f"{Fore.RED}The face is too far in the real image, far face{Style.RESET_ALL}")
    quit()
elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
    print(f"{Fore.RED}The face is too far in the real image, far face{Style.RESET_ALL}")
    quit()
elif ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error with the real image, far face{Style.RESET_ALL}")
    quit()

# Next, we need to obtain the 106 facial landmarks for the face. 
ret, far_landmarks = sdk.get_face_landmarks(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the real image, far face{Style.RESET_ALL}")
    quit()    

# Finally, we can compute a face recognition template for the face, 
# and later use it to ensure the two active spoof images are from the same identity.
ret, far_faceprint = sdk.get_face_feature_vector(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the real image, far face{Style.RESET_ALL}")
    quit()


# Now at this point, we can repeat all the above steps, but for the near image now. 
res = sdk.set_image("../images/near_shot_real_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set real image, near shot{Style.RESET_ALL}")
    quit()

image_props = sdk.get_image_properties()

found, fb = sdk.detect_largest_face()
if found == False:
    print(f"{Fore.RED}Unable to find face in real image, near shot{Style.RESET_ALL}")
    quit()

ret = sdk.check_spoof_image_face_size(fb, image_props, tfsdk.ACTIVESPOOFSTAGE.NEAR) # Be sure to specify near image this time.
if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
    print(f"{Fore.RED}The face is too far in the real image, near face{Style.RESET_ALL}")
    quit()
elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
    print(f"{Fore.RED}The face is too far in the real image, near face{Style.RESET_ALL}")
    quit()
elif ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error with the real image, near face{Style.RESET_ALL}")
    quit()

ret, near_landmarks = sdk.get_face_landmarks(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the real image, near face{Style.RESET_ALL}")
    quit()    

ret, near_faceprint = sdk.get_face_feature_vector(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the real image, near face{Style.RESET_ALL}")
    quit()

# At this point, we can finally run active spoof detection
ret, spoof_score, spoof_label = sdk.detect_active_spoof(near_landmarks, far_landmarks)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run active spoof detection for real images!{Style.RESET_ALL}")
    quit()

print("Printing results for real image.")

if spoof_label == tfsdk.SPOOFLABEL.FAKE:
    print("SPOOF RESULTS: Spoof attempt detected!")
else:
    # Finally, as a last step, we can compare the two face recognition templates to ensure they are of the same identity
    ret, match_prob, sim_score = sdk.get_similarity(near_faceprint, far_faceprint)
    if sim_score < 0.3:
        print("SPOOF RESULTS: Real image detected, but images are of different identities!")
    else:
        print("SPOOF RESULTS: Real image detected, and both images are of the same identity!")


# Now for the sake of the demo, let's repeat the entire process but with two face / spoof attempt images
# We will start with the far shot image
res = sdk.set_image("../images/far_shot_fake_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set fake image, far shot{Style.RESET_ALL}")
    quit()


# Next, we must obtain the image properties
# These properties are used by the check_spoof_image_face_size() function
image_props = sdk.get_image_properties()


# Next, must detect if there is a face in the image
found, fb = sdk.detect_largest_face()
if found == False:
    print(f"{Fore.RED}Unable to find face in fake image, far shot{Style.RESET_ALL}")
    quit()

# Now, we must ensure that the face meets the size criteria
# Be sure to check the return value of this function

ret = sdk.check_spoof_image_face_size(fb, image_props, tfsdk.ACTIVESPOOFSTAGE.FAR)
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
ret, far_landmarks = sdk.get_face_landmarks(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the fake image, far face{Style.RESET_ALL}")
    quit()    

# Finally, we can compute a face recognition template for the face, 
# and later use it to ensure the two active spoof images are from the same identity.
ret, far_faceprint = sdk.get_face_feature_vector(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the fake image, far face{Style.RESET_ALL}")
    quit()


# Now at this point, we can repeat all the above steps, but for the near image now. 
res = sdk.set_image("../images/near_shot_fake_person.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set real image, near shot{Style.RESET_ALL}")
    quit()

image_props = sdk.get_image_properties()

found, fb = sdk.detect_largest_face()
if found == False:
    print(f"{Fore.RED}Unable to find face in fake image, near shot{Style.RESET_ALL}")
    quit()

ret = sdk.check_spoof_image_face_size(fb, image_props, tfsdk.ACTIVESPOOFSTAGE.NEAR) # Be sure to specify near image this time.
if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
    print(f"{Fore.RED}The face is too far in the fake image, near face{Style.RESET_ALL}")
    quit()
elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
    print(f"{Fore.RED}The face is too far in the fake image, near face{Style.RESET_ALL}")
    quit()
elif ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error with the fake image, near face{Style.RESET_ALL}")
    quit()

ret, near_landmarks = sdk.get_face_landmarks(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error obtaining the facial landmarks for the fake image, near face{Style.RESET_ALL}")
    quit()    

ret, near_faceprint = sdk.get_face_feature_vector(fb)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}There was an error generating the feature vector for the fake image, near face{Style.RESET_ALL}")
    quit()

# At this point, we can finally run active spoof detection
ret, spoof_score, spoof_label = sdk.detect_active_spoof(near_landmarks, far_landmarks)
if ret != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to run active spoof detection for fake images!{Style.RESET_ALL}")
    quit()

print("Printing results for fake image.")

if spoof_label == tfsdk.SPOOFLABEL.FAKE:
    print("SPOOF RESULTS: Spoof attempt detected!")
else:
    # Finally, as a last step, we can compare the two face recognition templates to ensure they are of the same identity
    ret, match_prob, sim_score = sdk.get_similarity(near_faceprint, far_faceprint)
    if sim_score < 0.3:
        print("SPOOF RESULTS: Real image detected, but images are of different identities!")
    else:
        print("SPOOF RESULTS: Real image detected, and both images are of the same identity!")