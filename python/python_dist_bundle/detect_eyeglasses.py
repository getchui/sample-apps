# Sample code: Run eye glasses detection on two images (one image with eye glasses, one image without eye glasses)

import tfsdk
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
# options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
# options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

# Load the image with glasses
res = sdk.set_image("../images/glasses.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()


# detect the largest face
found, face_bounding_box = sdk.detect_largest_face()
if found:
    # Run glasses detection
    res, glasses_label, glasses_score = sdk.detect_glasses(face_bounding_box)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run glasses detection with image 1{Style.RESET_ALL}")
        quit()

    # Expect the result to be "glasses"
    print(f"Face with glasses, predicted result: {glasses_label}")

else:
    print("No face detected in image")

# Load the non glasses face image
res = sdk.set_image("../images/headshot.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 2{Style.RESET_ALL}")
    quit()

# detect the largest face
found, face_bounding_box = sdk.detect_largest_face()
if found:
    # Run glasses detection
    res, glasses_label, glasses_score = sdk.detect_glasses(face_bounding_box)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run glasses detection with image 2{Style.RESET_ALL}")
        quit()


    # Expect the result to be "no glasses"
    print(f"Face without glasses, predicted result: {glasses_label}")

else:
    print("No face detected in image")
