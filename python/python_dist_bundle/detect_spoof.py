# Sample code: Run spoof detection on two images (one real image, one spoofed image)

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

# Load the real image
res = sdk.set_image("../images/real.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# Run spoof detection
found, face_bounding_box = sdk.detect_largest_face()
if found:
    res, spoof_label, spoof_score = sdk.detect_spoof(face_bounding_box)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run spoof with image 1{Style.RESET_ALL}")
        quit()

# Expect the result to be real
print(f"Real image predicted result: {spoof_label}")

# Load the spoof atempt image
res = sdk.set_image("../images/fake.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 2{Style.RESET_ALL}")
    quit()


# Run spoof detection
found, face_bounding_box = sdk.detect_largest_face()
if found:
    res, spoof_label, spoof_score = sdk.detect_spoof(face_bounding_box)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run spoof with image 1{Style.RESET_ALL}")
        quit()

# Expect the result to be fake
print(f"Spoof attempt image predicted result: {spoof_label}")
