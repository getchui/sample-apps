import cv2
import tfsdk
import os
import sys
from imutils import paths

def usage():
    print("Usage: {} <collection folder name> <input video filename>".format(sys.argv[0]))

if len(sys.argv) != 3:
    usage()
    sys.exit(1)

filepath = sys.argv[2]

# if filepath is just a number, it's the ID of a local camera like a USB camera
if filepath.isdigit():
    filepath = int(filepath)

collection_file = sys.argv[1]
images = sorted(list(paths.list_images(collection_file)))

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

labels = []
for image in images:
    labels.append(''.join(os.path.basename(image).split('.')[:-1]))



options = tfsdk.ConfigurationOptions()


options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE # Save the templates in an SQLITE database

# To use a PostgreSQL database
# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.POSTGRESQL


model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 
options.fr_model = model
options.models_path = "/home/seelaman/Workspace/trueface/trueface.base/3.6_9044/tfsdk_python3.6_v0.23.9044/trueface_sdk/download_models"

# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
options.enable_GPU = True 

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print("Invalid License Provided")
    print("Be sure to export your license token as TRUEFACE_TOKEN")
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


out = cv2.VideoWriter(os.path.join(out_directory, f"detected-{model}-{filename}"),
                      fourcc, 25.0, (out_width, out_height))



# Create a new database
res = sdk.create_database_connection(collection_file)
if (res != tfsdk.ERRORCODE.NO_ERROR):
  print("Unable to create database connection")
  quit()

# ex. If using POSTGRESQL backend...
# res = sdk.create_database_connection("host=localhost port=5432 dbname=my_database user=postgres password=admin")
# if (res != tfsdk.ERRORCODE.NO_ERROR):
#   print("Unable to create database connection")
#   quit()

# Create a new collection
res = sdk.create_load_collection("my_collection")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print("Unable to create collection")
    quit()


logo_img = cv2.imread("/home/seelaman/Pictures/trueface/Trueface_black_120_wide.png")
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

            oH,oW = frame.shape[:2]
            frame = np.dstack([frame, np.ones((oH,oW), dtype="uint8") * 255])

            #Resizing the image
            scl = 10
            w = int(logo_img.shape[1] * scl / 100)
            h = int(logo_img.shape[0] * scl / 100)
            dim = (w,h)
            logo = cv2.resize(logo_img, dim, interpolation = cv2.INTER_AREA)
            lH,lW = logo.shape[:2]

            #Blending
            ovr = np.zeros((oH,oW,4), dtype="uint8")
            ovr[oH - lH - 60:oH - 60, oW - lW - 10:oW - 10] = logo
            final = image.copy()
            final = cv2.addWeighted(ovr,0.5,final,1.0,0,final)
            # ShoWing the result
            frame = final
            #cv2.imshow("Combine Image",final)



#            y1, y2 = int(facebox.bottom_right.y), int(facebox.bottom_right.y) + logo.shape[0]
#            x1, x2 = int(facebox.top_left.x), int(facebox.top_left.x) + logo.shape[1]
#            alpha_s = logo[:, :, 2] / 255.0
#            alpha_l = 1.0 - alpha_s
#
#            for c in range(0, 2):
#                frame[y1:y2, x1:x2, c] = (alpha_s * logo[:, :, c] +
#                                          alpha_l * frame[y1:y2, x1:x2, c])
#            draw_label(frame,
#                       (int(facebox.top_left.x), int(facebox.bottom_right.y)),
#                       "{} {}%".format(
#                           candidate.identity,
#                           int(candidate.match_probability*100)))

    out.write(frame)
    cv2.imshow(filename, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()

