# Sample code: Get frame from webcam, run object detection on the frame,
# then display the frame
# Note: you will need to have the opencv-python module installed

import cv2
import os
import sys
import tfsdk
import pafy
import datetime


def draw_label(image, point, label,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.0, thickness=2):
    """Draw label on the image
    Args:
        image (str or binary): image path, base64 encoded image, numpy array
        or OpenCV image
        point (tuple): (x_label, y_label)
        label (str): The label you want to write
        font (int): your preferred font
        font_scale (float): scaling factor for the font
        thickness (int): thickness of text
    """
    cv2.putText(
        image, label.capitalize(), point, font, font_scale,
        (0, 255, 0), thickness)


def draw_rectangle(frame, bounding_box):
    # Draw the rectangle on the frame
    cv2.rectangle(frame,
                  (int(bounding_box.top_left.x), int(bounding_box.top_left.y)),
                  (int(bounding_box.bottom_right.x),
                   int(bounding_box.bottom_right.y)),
                  (255, 0, 255), 3)


def usage():
    print("Usage: {} <collection folder name> "
          "<input video filename>".format(sys.argv[0]))


if len(sys.argv) != 3:
    usage()
    sys.exit(1)
collection_folder = sys.argv[1]
filepath = sys.argv[2]
if filepath.isdigit():
    filepath = int(filepath)

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "rtsp_transport;udp"

if isinstance(filepath, int):
    out_directory = os.path.curdir
    filename = "camera{}.mp4".format(filepath)
elif 'youtube' in filepath:
    video = pafy.new(filepath)
    best = video.getbest(preftype="mp4")
    filepath = best.url
    out_directory = '.'
    filename = best.filename
elif 'rtsp://' in filepath:
    out_directory = '.'
    filename = f"{datetime.datetime.now().strftime('%Y%m%d.%H:%M:%S')}.rtsp.stream.h264.mp4"
else:
    out_directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

# Use the default camera
cap = cv2.VideoCapture(filepath)

# VideoWriter
scale_factor = 0.5
res, frame = cap.read()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

scaled_width = int(width * scale_factor)
scaled_height = int(height * scale_factor)
# out_width = 640
# out_height = 480

# Rotated
# out_width = height
# out_height = width

# not rotated
out_width = scaled_width
out_height = scaled_height

options = tfsdk.ConfigurationOptions()
options.enable_GPU = True
options.GPU_device_index = 0
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.FULL
options.fd_filter = tfsdk.FACEDETECTIONFILTER.HIGH_PRECISION
options.fr_vector_compression = True
options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE
options.smallest_face_height = int(scaled_height / 10)
options.models_path = "/home/camarografiatrueface/Descargas/cuda6.0.fdchange/tfsdk_python3.7/trueface_sdk"
# options.smallest_face_height = 100

sdk = tfsdk.SDK(options)

# TODO: replace the string with your license code.
is_valid = sdk.set_license(os.environ['TOKEN'])
if not is_valid:
    message = "Invalid License Provided"
    quit()

collection_file = "my_database.db"
sdk.create_database_connection(collection_file)
res = sdk.create_load_collection("my_collection")

#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fourcc = cv2.VideoWriter_fourcc(*'X264')

out = cv2.VideoWriter(os.path.join(out_directory,
                                   "detected-FULL-{}".format(filename)),
                      fourcc, 25.0, (out_width, out_height))

# Create a new collection
# build the collection
# Some identities to populate into our collection
# read folder argument
# make a list of  tuples of the form (image_path, label)
# sdk.create_database_connection(collection_file)
# res = sdk.create_load_collection("my_collection")
# images = sorted(list(paths.list_images(collection_folder)))
# for image in images:
#    print("setting {}".format(image))
#    res = sdk.set_image(image)
#    if (res != tfsdk.ERRORCODE.NO_ERROR):
#        print("Couldn't set image for {}".format(image))
#        continue
#    try:
#        res, faceprint = sdk.get_largest_face_feature_vector()
#    except Exception as e:
#        print(e)
#        continue
#
#    if (res != tfsdk.ERRORCODE.NO_ERROR):
#        print("Couldn't get largest feature vector for {}".format(image))
#        continue
#    print("enrolling {}".format(image))
#    sdk.enroll_template(faceprint, os.path.split(os.path.dirname(image))[-1])


counter = 0
frameskip = 2

faceboxes = []
candidate = None

# track if we previously detected a face because sdk.get_face_feature_vector()
# is very slow on first call takes over 10s to load into memory. Until that
# has run for the first time, don't show any output so it doesn't look laggy
first_face_detected = False
identities = {}

logfile_name = f"{sys.argv[0]}." \
               f"{datetime.datetime.now().strftime('%Y%m%d.%H:%M:%S')}.log"
logfile = open(logfile_name, 'w')
logfile.write(f"timestamp,match,identity,similarity_measure,"
              f"match_probability,message\n")
while True:
    res, frame = cap.read()
    if not res:
        break
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    frame = cv2.resize(frame, (scaled_width, scaled_height), cv2.INTER_AREA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # print(f"counter: {counter}")
    if counter % frameskip > 0:
        for identity in identities:
            facebox = identities[identity]['facebox']
            draw_rectangle(frame, facebox)
            draw_label(frame,
                       (int(facebox.top_left.x), int(facebox.top_left.y)),
                       "{} {}%".format(
                           identities[identity]['label'],
                           int(identities[identity][
                                   'match_probability'] * 100)))
        cv2.putText(frame, timestamp, (20, 20), font, 1, (255, 255, 255), 2)
        out.write(frame)
        if first_face_detected:
            cv2.imshow(filename, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        counter += 1
        continue
    # Capture frame-by-frame
    counter += 1

    # frame = cv2.flip(frame, 0)
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Set the image using the frame buffer. OpenCV stores images in BGR format
    res = sdk.set_image(frame, scaled_width, scaled_height, tfsdk.COLORCODE.bgr)
    if res != tfsdk.ERRORCODE.NO_ERROR:
        message = f"Unable to set frame."
        logfile.write(f"{timestamp},False,,,,{message}\n")
        logfile.flush()

        cv2.putText(frame, timestamp, (20, 20), font, 1, (255, 255, 255), 2)
        out.write(frame)
        if first_face_detected:
            cv2.imshow(filename, frame)
        continue

    faceboxes = []
    faceboxes = sdk.detect_faces()
    if not faceboxes or len(faceboxes) == 0:
        message = "Unable to detect faces."
        identities = {}
        logfile.write(f"{timestamp},False,,,,{message}\n")
        logfile.flush()
        cv2.putText(frame, timestamp, (20, 20), font, 1, (255, 255, 255), 2)
        out.write(frame)
        if first_face_detected:
            cv2.imshow(filename, frame)
        continue

    # Run 1:N search for all extracted faces
    identities = {}
    faceboxes = [facebox for facebox in faceboxes if facebox.score > 0.961]

    for i, facebox in enumerate(faceboxes):
        res, faceprint = sdk.get_face_feature_vector(facebox)
        identities[i] = {}
        identities[i]['facebox'] = facebox
        if res != tfsdk.ERRORCODE.NO_ERROR:
            message = "skipping facebox"
            identities[i]['label'] = "Skipping"
            logfile.write(f"{timestamp},False,,,,{message}\n")
            logfile.flush()
            continue
        first_face_detected = True
        res, match_bool, candidate = sdk.identify_top_candidate(faceprint,
                                                                threshold=0.5)
        draw_rectangle(frame, facebox)
        if res != tfsdk.ERRORCODE.NO_ERROR:
            message = "Error identifying match"
            identities[i]['label'] = "id error"
            logfile.write(f"{timestamp},False,,,,{message}\n")
            logfile.flush()
            continue
        elif not match_bool:
            message = "No matching face in collection"
            logfile.write(f"{timestamp},False,,,,{message}\n")
            logfile.flush()
            identities[i]['label'] = "Unknown"
            identities[i]['match_probability'] = 0
            identities[i]['similarity_measure'] = 0
            draw_label(frame,
                       (int(facebox.top_left.x), int(facebox.top_left.y)),
                       "{} {}%".format(
                           identities[i]['label'],
                           int(identities[i]['match_probability'] * 100)))
            logfile.write(
                f"{timestamp},{match_bool},{identities[i]['label']},"
                f"{identities[i]['similarity_measure']},"
                f"{identities[i]['match_probability']},\n")
            logfile.flush()

            continue
        elif match_bool:
            message = "{} sim: {:.2f} prob: {}%".format(
                candidate.identity,
                candidate.similarity_measure,
                int(candidate.match_probability * 100))

            identities[i]['label'] = candidate.identity
            identities[i]['match_probability'] = candidate.match_probability
            draw_label(frame,
                       (int(facebox.top_left.x), int(facebox.top_left.y)),
                       "{} {}%".format(
                           identities[i]['label'],
                           int(identities[i]['match_probability'] * 100)))
            logfile.write(
                f"{timestamp},{match_bool},{candidate.identity},"
                f"{candidate.similarity_measure},"
                f"{candidate.match_probability},\n")
            logfile.flush()

    cv2.putText(frame, timestamp, (20, 20), font, 1, (255, 255, 255), 2)
    out.write(frame)
    cv2.imshow(filename, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
logfile.close()
