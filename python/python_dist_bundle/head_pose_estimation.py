# Sample code: Compute the yaw pitch and roll of a face image

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

# Load the input image
res = sdk.set_image("../images/brad_pitt_1.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# detect the largest face
found, face_bounding_box = sdk.detect_largest_face()

if found == False:
    print(f"{Fore.RED}Unable to detect face{Style.RESET_ALL}")
    quit()

res, yaw, pitch, roll = sdk.estimate_head_orientation(face_bounding_box)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to compute orientation{Style.RESET_ALL}")
    quit();

print(f'yaw: {yaw} radians')
print(f'pitch: {pitch} radians')
print(f'roll: {roll} radians')



