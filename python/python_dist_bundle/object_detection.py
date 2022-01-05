# Sample code: Run object detection on image and print the results

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

# Use the accurate object detector
options.obj_model = tfsdk.OBJECTDETECTIONMODEL.ACCURATE 

# Since we know we will use object detection,
# we can choose to initialize this modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.object_detector = True
options.initialize_module = initializeModule

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

# Load the input image of a person on a bike
res = sdk.set_image("../images/person_on_bike.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# Run object detection
bounding_boxes = sdk.detect_objects()
if (len(bounding_boxes) == 0) :
    print(f"{Fore.RED}Unable to find any objects in image{Style.RESET_ALL}")
    quit()


for bounding_box in bounding_boxes:
    # Get the bounding box label as a string
    label_string = sdk.get_object_label_string(bounding_box.label)


    print(f"Detected: {label_string} with probability: {bounding_box.probability}")
    print(f"Top left: ({bounding_box.top_left.x}, {bounding_box.top_left.y}), width: {bounding_box.width}, height: {bounding_box.height}")
    print()


