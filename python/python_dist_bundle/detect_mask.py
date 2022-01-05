# Sample code: Run mask detection on two images (one image with mask, one image without mask)

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

# Load the image with mask
res = sdk.set_image("../images/mask.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# detect the largest face
found, face_bounding_box = sdk.detect_largest_face()
if found:
    # Run mask detection
    res, mask_label = sdk.detect_mask(face_bounding_box)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run mask detection with image 1{Style.RESET_ALL}")
        quit()

    # Expect the result to be "mask"
    print(f"Face with mask, predicted result: {mask_label}")

else:
    print("No face detected in image")

# Load the non mask face image
res = sdk.set_image("../images/headshot.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 2{Style.RESET_ALL}")
    quit()

# detect the largest face
found, face_bounding_box = sdk.detect_largest_face()
if found:
    # Run mask detection
    res, mask_label = sdk.detect_mask(face_bounding_box)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run mask detection with image 2{Style.RESET_ALL}")
        quit()


    # Expect the result to be "no mask"
    print(f"Face without mask, predicted result: {mask_label}")

else:
    print("No face detected in image")