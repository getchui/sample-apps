import tfsdk
import cv2
import os
from colorama import Fore
from colorama import Style
from pynput import keyboard

def draw_rectangle(frame, bounding_box, color):

    # Draw the rectangle on the frame
    cv2.rectangle(frame,
                  (int(bounding_box.top_left.x), int(bounding_box.top_left.y)),
                  (int(bounding_box.bottom_right.x), int(bounding_box.bottom_right.y)), color, 3)

def show_frame(frame):
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True
    return False
        
def draw_near_ellipse(frame, color):
        center_x = frame.shape[1]//2
        center_y = frame.shape[0]//2
        length = frame.shape[0]//4
        width = length*2//3
        cv2.ellipse(frame, 
            (center_x, center_y), 
            (width, length), 
            0, 0, 360, 
            color, 
            2)

def draw_far_ellipse(frame, color):
    center_x = frame.shape[1]//2
    center_y = frame.shape[0]//2
    length = frame.shape[0]//2
    width = length*2//3
    cv2.ellipse(frame, 
        (center_x, center_y), 
        (width, length), 
        0, 0, 360, 
        color, 
        2)

def draw_text(frame, text, color, location=(40, 40)):
    cv2.putText(frame, text, 
        location, 
        font, 
        font_scale,
        color,
        text_width,
        cv2.LINE_AA)

# Event listener for the spacebar
g_spacebar_pressed = False

def on_press(key):
    if key == keyboard.Key.esc:
        return False  # stop listener
    if key == keyboard.Key.space:
         global g_spacebar_pressed
         if not g_spacebar_pressed:
            g_spacebar_pressed = True

listener = keyboard.Listener(
    on_press=on_press)
listener.start()


options = tfsdk.ConfigurationOptions()
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 # We will use TFV5 to verify that both images are from the same person

# options.enable_GPU = True # Use GPU inference to improve speed
# You will require the GPU SDK and GPU token for this.

options.fd_filter = tfsdk.FACEDETECTIONFILTER.HIGH_PRECISION # Use high precision filter to remove any low quality faces which will have inaccurate landmarks.

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

state = 0

font = cv2.FONT_HERSHEY_SIMPLEX 
font_scale = 1
text_width = 2
text_color = (255, 255, 100)

while(True):
    # Grab frame from camera
    ret, frame = cap.read()
    if ret == False:
        continue

    # Flip image horizontally
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Set the image using the frame buffer. OpenCV stores images in BGR format
    res = sdk.set_image(frame_copy, frame_copy.shape[1], frame_copy.shape[0], tfsdk.COLORCODE.bgr)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set frame{Style.RESET_ALL}")
        if show_frame(frame):
            break;
        continue

    # Next, we must obtain the image properties
    # These properties are used by the check_spoof_image_face_size() function
    image_props = sdk.get_image_properties()    

    if state == 0:
        draw_near_ellipse(frame, (0, 0, 255))

        # Need to capture far image
        found, fb = sdk.detect_largest_face()
        if found == False:
            # Unable to detect face in image
            if show_frame(frame):
                break
            continue

        ret = sdk.check_spoof_image_face_size(fb, image_props, tfsdk.ACTIVESPOOFSTAGE.FAR)

        if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
            draw_text(frame, "Move closer", (0, 0, 255))
            draw_rectangle(frame, fb, (0, 0, 255))
            if show_frame(frame):
                break
            continue

        elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
            draw_text(frame, "Move farther", (0, 0, 255))
            draw_rectangle(frame, fb, (0, 0, 255))
            if show_frame(frame):
                break;
            continue

        elif ret != tfsdk.ERRORCODE.NO_ERROR:
            print(f"{Fore.RED}Unable to run check_spoof_image_face_size() function{Style.RESET_ALL}")
            if show_frame(frame):
                break;
            continue

        draw_near_ellipse(frame, (0, 255, 0))
        draw_rectangle(frame, fb, (0, 255, 0))
        draw_text(frame, "Press space to capture image" , (0, 255, 0))

        if g_spacebar_pressed == True:
            g_spacebar_pressed = False
            # Obtain the face landmarks
            ret, far_landmarks = sdk.get_face_landmarks(fb)
            if ret != tfsdk.ERRORCODE.NO_ERROR:
                print(f"{Fore.RED}Unable to get face landmarks{Style.RESET_ALL}")
                if show_frame(frame) == True:
                    break;
                continue

            # Also generate a face recognition template so that we can both images are of the same identity
            ret, far_faceprint = sdk.get_face_feature_vector(fb)
            if ret != tfsdk.ERRORCODE.NO_ERROR:
                print(f"{Fore.RED}There was an error generate the face feautre vector{Style.RESET_ALL}")
                if show_frame(frame) == True:
                    break;
                continue

            # At this point, we can increment the state counter
            state = 1

    if state == 1:
        draw_far_ellipse(frame, (0, 0, 255))

        # Need to capture near image
        found, fb = sdk.detect_largest_face()
        if found == False:
            # Unable to detect face in image
            if show_frame(frame):
                break
            continue

        ret = sdk.check_spoof_image_face_size(fb, image_props, tfsdk.ACTIVESPOOFSTAGE.NEAR)

        if ret == tfsdk.ERRORCODE.FACE_TOO_FAR:
            draw_text(frame, "Move closer", (0, 0, 255))
            draw_rectangle(frame, fb, (0, 0, 255))
            if show_frame(frame):
                break
            continue

        elif ret == tfsdk.ERRORCODE.FACE_TOO_CLOSE:
            draw_text(frame, "Move farther", (0, 0, 255))
            draw_rectangle(frame, fb, (0, 0, 255))
            if show_frame(frame):
                break;
            continue

        elif ret != tfsdk.ERRORCODE.NO_ERROR:
            print(f"{Fore.RED}Unable to run check_spoof_image_face_size() function{Style.RESET_ALL}")
            if show_frame(frame):
                break;
            continue

        draw_far_ellipse(frame, (0, 255, 0))
        draw_rectangle(frame, fb, (0, 255, 0))
        draw_text(frame, "Press space to capture image" , (0, 255, 0))

        if g_spacebar_pressed == True:
            g_spacebar_pressed = False
            # Obtain the face landmarks
            ret, near_landmarks = sdk.get_face_landmarks(fb)
            if ret != tfsdk.ERRORCODE.NO_ERROR:
                print(f"{Fore.RED}Unable to get face landmarks{Style.RESET_ALL}")
                if show_frame(frame) == True:
                    break;
                continue

            # Also generate a face recognition template so that we can both images are of the same identity
            ret, near_faceprint = sdk.get_face_feature_vector(fb)
            if ret != tfsdk.ERRORCODE.NO_ERROR:
                print(f"{Fore.RED}There was an error generate the face feautre vector{Style.RESET_ALL}")
                if show_frame(frame) == True:
                    break;
                continue

            # Increment the state
            state = 2

        if state == 2:
            # Run the active spoof detection
            ret, spoof_score, spoof_label = sdk.detect_active_spoof(near_landmarks, far_landmarks)
            if ret != tfsdk.ERRORCODE.NO_ERROR:
                print(f"{Fore.RED}Unable to run active spoof, try again!{Style.RESET_ALL}")
                state = 0
                continue;

            state = 3

    if state == 3:
        if spoof_label == tfsdk.SPOOFLABEL.FAKE:
            draw_text(frame, "Spoof attempt detected!", (0, 0, 255))
        else:
            # Run face recognition to ensure the images are of the same identity
            ret, match_prob, sim_score = sdk.get_similarity(near_faceprint, far_faceprint)
            if sim_score < 0.3:
                draw_text(frame, "Images are not of the same identity!", (0, 0, 255))
            else:
                draw_text(frame, "Real face detected!" , (0, 255, 0))


        draw_text(frame, "Press the space bar to restart", (255, 0, 0), (40, 80))
        if g_spacebar_pressed == True:
            g_spacebar_pressed = False
            state = 0


            

    # Display the resulting frame
    if show_frame(frame):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

