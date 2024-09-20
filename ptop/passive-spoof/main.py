'''
An example app for demonstrating user-driven feedback to successfully
perform passive spoof detection.

requirements.txt:
opencv-python
numpy
requests
'''
import cv2
import numpy as np
import requests

import base64
from dataclasses import dataclass
import time

# Constants for color
COLOR_ERROR = (0, 0, 255)
COLOR_SUCCESS = (0, 255, 0)


@dataclass
class Point2D:
    '''
    A utility class for being an intermediate between PTOP's JSON
    return and cv2's shape tuple.
    '''
    x: int
    y: int


def draw_rectangle(
        frame: cv2.Mat,
        center: Point2D,
        shape: Point2D,
        color_code=COLOR_SUCCESS):
    '''
    Draw a `color_code` colored rectangle on the frame centered at
    `center` with the provide `shape`
    '''
    tl = Point2D(x=center.x-shape.x/2, y=center.y-shape.y/2)
    br = Point2D(x=center.x+shape.x/2, y=center.y+shape.y/2)

    cv2.rectangle(
            img=frame,
            pt1=(int(tl.x), int(tl.y)),
            pt2=(int(br.x), int(br.y)),
            color=color_code,
            thickness=3)


def draw_label(
        frame: cv2.Mat,
        point: Point2D,
        label: str,
        color_code=(194, 134, 58),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale=1.0, thickness=2):
    '''
    Draw a label on the frame.

    NOTE: point is the bottom left of the first letter!
    '''
    font_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x_label = int(point.x)
    y_label = int(point.y)

    for line in label.split('\n'):
        cv2.putText(
            frame, line.capitalize(), (x_label, y_label), font, font_scale,
            color_code, thickness, cv2.LINE_AA)
        y_label += font_size[1] + 10


def convert_mat_to_b64_encoded_jpg(frame: cv2.Mat):
    image_encode = cv2.imencode('.jpg', frame)[1]
    data_encode = np.array(image_encode)
    byte_encode = data_encode.tobytes()
    b64image = base64.b64encode(byte_encode)
    return b64image


def detect_faces(base_url: str, b64image: str, doSpoof: bool):
    '''
    Call the detect-face
    '''
    payload = {
        'image_base64': b64image,
        'face_landmarks_106': 'false',
        'largest_face_only': 'true',
        'detect_mask': 'false',
        'detect_spoof': 'true' if doSpoof else 'false'
    }

    url = f'{base_url}/v2/detect-faces'
    r = requests.post(url=url, data=payload)
    # NOTE: --no-seatbelts with status_code handling for demo, always check
    # status codes!

    return r.status_code, r.json()


def convert_ptop_error_to_display_message(ptop_error_message: str):
    '''
    A simple utility function to convert a PTOP error message to an
    informational message for an end-user.    
    '''
    display_msg = ''
    if "face not centered" in ptop_error_message:
        display_msg = "Place face in center of image"
    elif 'too dark' in ptop_error_message:
        display_msg = "Illuminate face"
    elif 'extreme face angle' in ptop_error_message:
        display_msg = "Look directly at camera"
    elif 'eyes closed' in ptop_error_message:
        display_msg = "Open eyes"
    elif 'face too close' in ptop_error_message:
        display_msg = 'Move back'
    elif 'face too small' in ptop_error_message:
        display_msg = 'Move closer'
    elif 'No face detected' in ptop_error_message:
        # this can happen if the person walks away from the computer
        display_msg = 'No face detected'
    else:
        # For other unhandled error conditions
        display_msg = ptop_error_message['message']

    return display_msg


def handle_error(frame: cv2.Mat, res: dict):
    '''
    The frame does satisfy the preconditions for spoof. Inform the user
    with an informational message on the frame so that they can correct
    '''

    display_msg = convert_ptop_error_to_display_message(res['message'])
    draw_label(frame, Point2D(50, 50), display_msg, color_code=COLOR_ERROR)


def get_face_info(res: dict) -> dict:
    '''
    Utility to extract the bounding box and spoof information from
    the PTOP response
    '''
    data = res['data']

    detected_face = data['detected_faces'][0]
    bounding_box = detected_face['bounding_box']
    top_left = Point2D(
        bounding_box['top_left']['x'],
        bounding_box['top_left']['y'])
    bottom_right = Point2D(
        bounding_box['bottom_right']['x'],
        bounding_box['bottom_right']['y'])

    spoof_label = detected_face['spoof']
    spoof_score = detected_face['spoof_score']

    face_info = {
        'spoof_label': spoof_label,
        'spoof_score': spoof_score,
        'bb': [top_left, bottom_right]
    }

    return face_info


def draw_bounding_box(
        frame: cv2.Mat, top_left: Point2D, bottom_right: Point2D):
    bb_shape = Point2D(
        bottom_right.x - top_left.x,  # width
        bottom_right.y - top_left.y)  # height
    bb_center = Point2D(
        x=top_left.x + bb_shape.x/2,  # left + width/2
        y=top_left.y + bb_shape.y/2,  # top + height/2
    )

    draw_rectangle(frame, bb_center, bb_shape, color_code=COLOR_SUCCESS)


def draw_spoof_info(
        frame: cv2.Mat, spoof_label: str, spoof_score: float):
    is_spoof = True if spoof_label == 'fake' else False
    success_msg = f'Spoof = {is_spoof}\nScore = {spoof_score:0.3f}'

    draw_label(
        frame,
        Point2D(25, 25),
        success_msg,
        color_code=COLOR_SUCCESS,
        font_scale=0.8)


def handle_success(frame: cv2.Mat, res: dict):
    '''
    The frame passed PTOP's pre-conditions. Provide feedback to user.
'
    NOTE: This does NOT mean that `spoof_label == 'real'`!
    '''
    face_info = get_face_info(res)

    # draw a bounding box around the face
    draw_bounding_box(frame, face_info['bb'][0], face_info['bb'][1])

    # display the returned spoof information
    draw_spoof_info(frame, face_info['spoof_label'], face_info['spoof_score'])


def set_camera_resolution(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def main():
    base_url = 'https://your.ptop.domain.or.ip'
    #
    # NOTE:
    #
    # Resolution plays a significant role in detecting spoof, but there
    # are tradeoffs:
    # * You want higher resolution for more captured information
    #
    # * A higher resolution implies
    #   + that the face will appear smaller, resulting in more
    #     "Move closer" messages
    #   + requires more processing (slower)
    #   + requires more bandwidth
    #
    # * A lower resolution implies
    #   + that the face will appear larger, resulting in more
    #     "Move back" messages
    #   + requires less processing overhead (faster)
    #   + requires less bandwidth
    #
    # * The "correct" resolution is one that strikes the balance between
    #   these factors, but 1280x768 is a good place to start.
    #
    cam = cv2.VideoCapture(0)
    frame_width, frame_height = set_camera_resolution(cam, 1280, 768)

    # Cameras are typically capable of at least 30fps, but the ability
    # of individuals to react to messages on the screen, e.g. Move back, or
    # face the camera, is slower. So 30fps is just wasted processing.
    #
    # This is just an example of how one could adjust the sampling rate.
    frame_rate = 15
    prev = 0

    doCapture = True
    while True:
        time_elapsed = time.time() - prev
        ret, frame = cam.read()
        if not ret:
            print('error reading from camera')
            break

        if time_elapsed > 1./frame_rate and doCapture:
            prev = time.time()

            status, res = detect_faces(base_url,
                                       convert_mat_to_b64_encoded_jpg(frame),
                                       doSpoof=True)
            # NOTE: success here implies that we were able to get a result
            # from PTOP, and NOT that it was spoofed/unspoofed
            if res['success'] is False:
                handle_error(frame, res)
            else:
                # the demo pauses the rendering of frames to simplify
                # user feedback.
                handle_success(frame, res)
                print('PAUSE capture')
                doCapture = False

            cv2.imshow('Camera', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            print('CONTINUE capture')
            doCapture = True

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
