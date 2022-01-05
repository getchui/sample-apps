# Sample code: Determine the body pose

import tfsdk
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
# options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
# options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.FULL

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()


# Load the input image
res = sdk.set_image("../images/person_on_bike.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# Run object detection on the image
bounding_boxes = sdk.detect_objects()

if len(bounding_boxes) == 0:
    print(f"{Fore.RED}Unable to detect objects in image{Style.RESET_ALL}")
    quit()

# Do not need to check if the detected object is a person, the estimate_pose function will do that for us
body_landmarks = sdk.estimate_pose(bounding_boxes)

# Now draw the pose on the image and save the file to disk
sdk.draw_pose("pose_image", body_landmarks)





