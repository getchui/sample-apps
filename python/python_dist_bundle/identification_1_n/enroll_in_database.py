# Sample code: Generate face recognition templates for images and then enroll them into a collection.

# This sample app demonstrates how you can enroll face recognition templates or Faceprints into a collection on disk.
# First, we create a database and create a new collection within that database.
# Next, we generate face recognition templates and enroll those templates into the collection.
# Note, after running this sample app, you can run the identification_1_n sample app.
import tfsdk
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
# Can set configuration options here

options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE # Save the templates in an SQLITE database

# To use a PostgreSQL database
# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.POSTGRESQL

options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.GPU_options = True

# To enable database encryption... 
# encryptDatabase = tfsdk.EncryptDatabase()
# encryptDatabase.enable_encryption = True
# encryptDatabase.key = "TODO: Your encryption key here"
# options.encrypt_database = encryptDatabase

# Since we know we will use the face detector and face recognizer,
# we can choose to initialize these modules in the SDK constructor instead of using lazy initialization
initializeModule = tfsdk.InitializeModule()
initializeModule.face_detector = True
initializeModule.face_recognizer = True
options.initialize_module = initializeModule

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

# Create a new database
res = sdk.create_database_connection("my_database.db")
if (res != tfsdk.ERRORCODE.NO_ERROR):
  print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
  quit()

# ex. If using POSTGRESQL backend...
# res = sdk.create_database_connection("host=localhost port=5432 dbname=my_database user=postgres password=admin")
# if (res != tfsdk.ERRORCODE.NO_ERROR):
#   print("Unable to create database connection")
#   quit()

# Create a new collection
res = sdk.create_load_collection("my_collection")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to create collection{Style.RESET_ALL}")
    quit()

# Since our collection is empty, lets populate the collection with some identities
image_identities = [
    ("../images/brad_pitt_2.jpg", "Brad Pitt"),
    ("../images/brad_pitt_3.jpg", "Brad Pitt"), # Can add the same identity more than once
    ("../images/tom_cruise_1.jpg", "Tom Cruise")
]

for path, identity in image_identities:
    print("Processing image:", path, "with identity:", identity)
    # Generate a template for each image
    res = sdk.set_image(path)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set image at path: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # Detect the largest face in the image
    found, faceBoxAndLandmarks = sdk.detect_largest_face()
    if found == False:
        print(f"{Fore.RED}No face detected in image: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # We want to only enroll high quality images into the database / collection
    # For more information, refer to the section titled "Selecting the Best Enrollment Images"
    # https://reference.trueface.ai/cpp/dev/latest/py/identification.html

    # Therefore, ensure that the face height is at least 100px
    faceHeight = faceBoxAndLandmarks.bottom_right.y - faceBoxAndLandmarks.top_left.y
    print(f"Face height: {faceHeight} pixels")

    if faceHeight < 100:
        print(f"{Fore.RED}The face is too small in the image for a high quality enrollment, not enrolling{Style.RESET_ALL}")
        continue

    # Get the aligned chip so we can compute the image quality
    face = sdk.extract_aligned_face(faceBoxAndLandmarks)

    # Compute the image quality score
    res, quality = sdk.estimate_face_image_quality(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}There was an error computing the image quality, not enrolling{Style.RESET_ALL}")
        continue

    # Ensure the image quality is above a threshold
    print("Face quality:", quality)
    if quality < 0.999:
        print(f"{Fore.RED}The image quality is too poor for enrollment, not enrolling{Style.RESET_ALL}")
        continue

    # We can check the orientation of the head and ensure that it is facing forward
    # To see the effect of yaw and pitch on match score, refer to: https://reference.trueface.ai/cpp/dev/latest/py/face.html#tfsdk.SDK.estimate_head_orientation

    res, yaw, pitch, roll = sdk.estimate_head_orientation(faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to compute head orientation, not enrolling{Style.RESET_ALL}")
        continue

    yaw_deg = yaw * 180 / 3.14
    pitch_deg = pitch * 180 / 3.14

    if abs(yaw_deg) > 50:
        print(f"{Fore.RED}Enrollment image has too extreme a yaw, not enrolling{Style.RESET_ALL}")
        continue

    if abs(pitch_deg) > 35:
        print(f"{Fore.RED}Enrollment image has too extreme a pitch, not enrolling{Style.RESET_ALL}")
        continue

    # Finally ensure the user is not wearing a mask
    error_code, mask_label = sdk.detect_mask(faceBoxAndLandmarks)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to run mask detection, not enrolling{Style.RESET_ALL}")
        continue

    if (mask_label == tfsdk.MASKLABEL.MASK):
        print(f"{Fore.RED}Please choose a image without a mask for enrollment, not enrolling{Style.RESET_ALL}")
        continue

    # Now that we have confirmed the images are high quality, generate a template from that image
    res, faceprint = sdk.get_face_feature_vector(face)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}There was an error generating the faceprint, not enrolling{Style.RESET_ALL}")
        continue

    # Enroll the feature vector into the collection
    res, UUID = sdk.enroll_faceprint(faceprint, identity)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to enroll feature vector{Style.RESET_ALL}")
        continue

    # TODO: Can store the UUID for later use
    print(f"Success, enrolled template with UUID: {UUID}")
    print("--------------------------------------------")
    print()


# For the sake of the demo, print the information about the collection
res, collection_names = sdk.get_collection_names()
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to get collection names{Style.RESET_ALL}")
    quit()

for collection_name in collection_names:
    res, metadata = sdk.get_collection_metadata(collection_name)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to get collection metadata{Style.RESET_ALL}")
        quit()

    print(f"Metadata: {metadata}")

    res, identities = sdk.get_collection_identities(collection_name)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to get collection identities{Style.RESET_ALL}")
        quit()

    print(f"Identities: {identities}")