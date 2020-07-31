# Sample code: Get frame from webcam, run object detection on the frame, then display the frame
# Note: you will need to have the opencv-python module installed

import cv2
import os
import sys
import tfsdk

from imutils import paths
from scipy import ndimage

def draw_label(image, point, label,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1.0, thickness=2):
    """Draw label on the image
    Args:
        img (str or binary): image path, base64 encoded image, numpy array or OpenCV image
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
                  (int(bounding_box.bottom_right.x), int(bounding_box.bottom_right.y)),
                 (255,0,255), 3)
                  #(255, 0, 255), 3)

def write_and_show(videowriter, frame):
    out.write(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)

def usage():
    print("Usage: {} <collection folder name> <input video filename>".format(sys.argv[0]))

if len(sys.argv) != 3:
    usage()
    sys.exit(1)
collection_folder = sys.argv[1]
filepath = sys.argv[2]
if filepath.isdigit():
    filepath = int(filepath)


options = tfsdk.ConfigurationOptions()
# Can set configuration options here
# ex:
# options.smallest_face_height = 40
options.fd_mode = tfsdk.FACEDETECTIONMODE.VERSATILE
# options.fd_filter = tfsdk.FACEDETECTIONFILTER.BALANCED
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.FULL
options.models_path = '/home/seelaman/Workspace/trueface/trueface.base/tfsdk_python3.7.gpu/trueface_sdk'
options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE

# Use the accurate object detector
#options.obj_model = tfsdk.OBJECTDETECTIONMODEL.ACCURATE

sdk = tfsdk.SDK(options)

# TODO: replace the string with your license code.
is_valid = sdk.set_license(os.environ['TOKEN'])
if (is_valid == False):
    print("Invalid License Provided")
    quit()

# Use the default camera
cap = cv2.VideoCapture(filepath)

# VideoWriter
res, frame = cap.read()
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# out_width = 640
# out_height = 480

#Rotated
#out_width = height
#out_height = width

# not rotated
out_width = width
out_height = height

fourcc = cv2.VideoWriter_fourcc(*'XVID')

if not isinstance(filepath, int):
    out_directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
else:
    out_directory = os.path.curdir
    filename = "camera{}.mkv".format(filepath)


out = cv2.VideoWriter(os.path.join(out_directory, "detected-tfv4-{}".format(filename)),
                      fourcc, 25.0, (out_width, out_height))

# Create a new collection
collection_file = "my_collection.sqlite"
if os.path.exists(collection_file):
    sdk.create_database_connection(collection_file)
    res = sdk.create_load_collection("my_memory_collection")
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to create collection. Error: {}".format(res))
        quit()
else:
    # build the collection
    # Some identities to populate into our collection
    # read folder argument
    # make a list of  tuples of the form (image_path, label)
    images = sorted(list(paths.list_images(collection_folder)))
    for image in images:
        sdk.set_image(image)
        res, faceprint = sdk.get_largest_face_feature_vector()
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Couldn't get largest feature vector for {}".format(image))
            continue
        print("enrolling {}".format(image))
        sdk.enroll_template(faceprint, os.path.split(os.path.dirname(image))[-1])


while(True):
    # Capture frame-by-frame
    res, frame = cap.read()
    if not res:
        break

    #frame = cv2.flip(frame, 0)
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Set the image using the frame buffer. OpenCV stores images in BGR format
    res = sdk.set_image(frame, width, height, tfsdk.COLORCODE.bgr)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to set frame.")
        out.write(frame)
        cv2.imshow(filename, frame)
        continue
    faceboxes = sdk.detect_faces()
    if not faceboxes or len(faceboxes) == 0:
        print("Unable to detect faces.")
        out.write(frame)
        cv2.imshow(filename, frame)
        continue

#    res, faceprints = sdk.get_face_feature_vectors()
#    if (res != tfsdk.ERRORCODE.NO_ERROR):
#        print("Unable extract face feature vectors.")
#        out.write(frame)
#        cv2.imshow(filename, frame)
#        continue

    # Run 1:N search for all extracted faces
    for facebox in faceboxes:
        res, faceprint = sdk.get_face_feature_vector(facebox)
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("skipping facebox")
            continue
        res, match_bool, candidate = sdk.identify_top_candidate(faceprint, threshold=0.3)
        draw_rectangle(frame, facebox)
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Unable find a matching face in the collection.")
            continue
        elif match_bool:
            print("{} sim: {:.2f} prob: {}%".format(
                       candidate.identity,
                       candidate.similarity_measure,
                       int(candidate.match_probability*100)))

            draw_label(frame,
                       (int(facebox.top_left.x), int(facebox.top_left.y)),
                       "{} {}%".format(
                           candidate.identity,
                           int(candidate.match_probability*100)))
            draw_label(frame,
                       (int(facebox.top_left.x), int(facebox.bottom_right.y)),
                       "{} {}%".format(
                           candidate.identity,
                           int(candidate.match_probability*100)))

    out.write(frame)
    cv2.imshow(filename, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()




