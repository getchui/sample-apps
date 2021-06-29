import tfsdk
import cv2
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 # We will use TFV5 to verify that both images are from the same person

options.enable_GPU = True # Use GPU inference to improve speed
# You will require the GPU SDK and GPU token for this.

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
    # Grab frame from camera
    ret, frame = cap.read()
    if ret == False:
        continue


    center_x = frame.shape[1]//2
    center_y = frame.shape[0]//2
    length = frame.shape[0]//4
    width = length*2//3
    cv2.ellipse(frame, 
        (center_x, center_y), 
        (width, length), 
        0, 0, 360, 
        (0,255,0), 
        1)


    center_x = frame.shape[1]//2
    center_y = frame.shape[0]//2
    length = frame.shape[0]//2
    width = length*2//3
    cv2.ellipse(frame, 
        (center_x, center_y), 
        (width, length), 
        0, 0, 360, 
        (0,255,0), 
        text_width)

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

