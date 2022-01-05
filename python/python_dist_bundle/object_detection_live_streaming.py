# Sample code: Get frame from webcam, run object detection on the frame, then display the frame
# Note: you will need to have the opencv-python module installed

import tfsdk
import cv2
import os
import time
from colorama import Fore
from colorama import Style

def draw_label(image, point, label,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.0, thickness=1):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x_label, y_label = point
    cv2.rectangle(
        image,
        (x_label, y_label - size[1] - 10),
        (x_label + size[0], y_label),
        (194,134,58),
        cv2.FILLED)

    cv2.putText(
        image, label.capitalize(), (x_label, y_label - 5), font, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA)


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


# Use the default camera (TODO: Can change the camera source, for example to an RTSP stream)
cap = cv2.VideoCapture(0)
if (cap.isOpened()== False): 
    print(f"{Fore.RED}Error opening video stream{Style.RESET_ALL}")
    os._exit(1)


while(True):
    # To skip some frames, uncomment the following
    # cap.grab()
    
    ret, frame = cap.read()
    if ret == False:
        continue


    # Set the image using the frame buffer. OpenCV stores images in BGR format
    res = sdk.set_image(frame, frame.shape[1], frame.shape[0], tfsdk.COLORCODE.bgr)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set frame.{Style.RESET_ALL}")
        continue

    # Run object detection
    bounding_boxes = sdk.detect_objects()
    if (len(bounding_boxes) > 0) :
        
        # Draw the bounding boxes and label on the frame
        for bounding_box in bounding_boxes:
            tl = (int(bounding_box.top_left.x), int(bounding_box.top_left.y))
            br = (int(bounding_box.top_left.x + bounding_box.width), int(bounding_box.top_left.y + bounding_box.height))

            # Draw the rectangle on the frame
            cv2.rectangle(frame, tl, br, (194,134,58), 3)

            label_string = sdk.get_object_label_string(bounding_box.label)
            draw_label(frame, tl, label_string)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()




