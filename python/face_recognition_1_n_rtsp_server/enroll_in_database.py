import tfsdk
import os
options = tfsdk.ConfigurationOptions()


options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE # Save the templates in an SQLITE database

# To use a PostgreSQL database
# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.POSTGRESQL

options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.enable_GPU = True 

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print("Invalid License Provided")
    print("Be sure to export your license token as TRUEFACE_TOKEN")
    quit()

# Create a new database
res = sdk.create_database_connection("my_database.db")
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

# Since our collection is empty, lets populate the collection with some identities
image_identities = [
    ("../../images/armstrong/armstrong1.jpg", "Armstrong"),
    ("../../images/obama/obama1.jpg", "Obama")
]

for path, identity in image_identities:
    print("Processing image:", path, "with identity:", identity)
    # Generate a template for each image
    res = sdk.set_image(path)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to set image at path:", path)
        continue

    # Detect the largest face in the image
    found, faceBoxAndLandmarks = sdk.detect_largest_face()
    if found == False:
        print("No face detected in image:", path)
        continue

    # We want to only enroll high quality images into the database / collection
    # For more information, refer to the section titled "Selecting the Best Enrollment Images"
    # https://reference.trueface.ai/cpp/dev/latest/py/identification.html

    # Therefore, ensure that the face height is at least 100px
    faceHeight = faceBoxAndLandmarks.bottom_right.y - faceBoxAndLandmarks.top_left.y
    print("Face height:", faceHeight, "pixels")

    if faceHeight < 100:
        print("The face is too small in the image for a high quality enrollment, not enrolling")
        continue

    # Get the aligned chip so we can compute the image quality
    face = sdk.extract_aligned_face(faceBoxAndLandmarks)
    
    # Compute the image quality score
    res, quality = sdk.estimate_face_image_quality(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error computing the image quality")
        continue

    # Ensure the image quality is above a threshold
    print("Face quality:", quality)
    if quality < 0.8:
        print("The image quality is too poor for enrollment, not enrolling")
        continue

    # We can check the orientation of the head and ensure that it is facing forward
    # To see the effect of yaw and pitch on match score, refer to: https://reference.trueface.ai/cpp/dev/latest/py/face.html#tfsdk.SDK.estimate_head_orientation

    res, yaw, pitch, roll = sdk.estimate_head_orientation(faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to compute head orientation")
        continue

    yaw_deg = yaw * 180 / 3.14
    pitch_deg = yaw * 180 / 3.14

    if abs(yaw_deg) > 50:
        print("Enrollment image has too extreme a yaw, not enrolling")
        continue

    if abs(pitch_deg) > 35:
        print("Enrollment image has too extreme a pitch, not enrolling")
        continue

    # Finally ensure the user is not wearing a mask
    error_code, mask_label, score = sdk.detect_mask(faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to run mask detection")
        continue

    if (mask_label == tfsdk.MASKLABEL.MASK):
        print("Please choose a image without a mask for enrollment, not enrolling")
        continue

    # Now that we have confirmed the images are high quality, generate a template from that image
    res, faceprint = sdk.get_face_feature_vector(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error generating the faceprint")
        continue

    # Enroll the feature vector into the collection
    res, UUID = sdk.enroll_faceprint(faceprint, identity)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to enroll feature vector")
        continue

    # TODO: Can store the UUID for later use
    print("Success, enrolled template with UUID:", UUID)
    print("--------------------------------------------")
    print()


# For the sake of the demo, print the information about the collection
res, collection_names = sdk.get_collection_names()
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print("Unable to get collection names")
    quit()

for collection_name in collection_names:
    res, metadata = sdk.get_collection_metadata(collection_name)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to get collection metadata")
        quit()

    print("Metadata: ", metadata)

    res, identities = sdk.get_collection_identities(collection_name)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to get collection identities")
        quit()

    print("Identities: ", identities)