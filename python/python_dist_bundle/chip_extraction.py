# Sample code: Extract and display the aligned 112x112 pixel face image

import tfsdk
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
# options.fd_mode = tfsdk.VERSATILE
# options.fd_filter = tfsdk.BALANCED
# options.fr_model = tfsdk.TFV5
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# Since we know we will use face detection,
# we can choose to initialize this modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.face_detector = True
options.initialize_module = initializeModule

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

if found:
    face = sdk.extract_aligned_face(face_bounding_box)

    # the extracted image is the RGB color format
    # opencv expects BGR
    import cv2
    face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    cv2.imshow('Aligned face chip', face_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No face detected in image")