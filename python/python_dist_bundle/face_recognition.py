# Sample code: Load two images from disk, extract feature vectors and compare similarity

import tfsdk
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
# options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.GPU_options = True

# Since we know we will use the face detector and face recognizer,
# we can choose to initialize these modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.face_detector = True
initializeModule.face_recognizer = True
options.initialize_module = initializeModule

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

# Set the second iamge
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


