# Sample code: Generate face recognition templates for images in batch and then enroll them into a collection.

# This sample app demonstrates how you can enroll face recognition templates or Faceprints into a collection on disk in batch.
# We take advantage of GPU batching to increase the template generation throughput.
# First, we create a database and create a new collection within that database.
# Next, we generate face recognition templates in batch and enroll those templates into the collection.
# Note, after running this sample app, you can run the identification_1_n sample app.

# For this sample app, we will assume the image name is the identity we want to enroll. 

import tfsdk
import os
import glob
from colorama import Fore
from colorama import Style

def generate_feature_vectors(face_chips, face_identities):
    res, faceprints = sdk.get_face_feature_vectors(face_chips)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to extract face feature vectors{Style.RESET_ALL}")
        quit()

    # Now enroll the faceprints into our collection
    for i in range(len(face_chips)):
        res, UUID = sdk.enroll_faceprint(faceprints[i], face_identities[i])
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print(f"{Fore.RED}Unable to enroll feature vector{Style.RESET_ALL}")
            continue

    print("================================")
    print("Enrolled batch of:", len(face_chips), "Faceprints in collection")
    print("================================")
    face_chips.clear()
    face_identities.clear()

options = tfsdk.ConfigurationOptions()
# Can set configuration options here

options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE # Save the templates in an SQLITE database

# To use a PostgreSQL database
# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.POSTGRESQL

options.smallest_face_height = 100

options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

options.GPU_options = True  # Batching is only supported by GPU

# TODO: Choose a batch size based on your GPU memory
# Consult our benchmarks page: https://docs.trueface.ai/Benchmarks-0b648f5a0cb84badb6425a12697a15e5
batch_size = 32

options.GPU_options.face_recognizer_GPU_options.opt_batch_size = batch_size
options.GPU_options.face_recognizer_GPU_options.max_batch_size = batch_size

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
  # print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
  # quit()

# Create a new collection
res = sdk.create_load_collection("my_collection")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to load collection{Style.RESET_ALL}")
    quit()


# TODO: You should change the following to point to the directory containing your images.
image_paths = glob.glob("/home/cyrus/Downloads/69000/*.png")

face_chips = []
face_identities = []


img_num = 0
for path in image_paths:
    img_num+=1
    if (img_num % 50 == 0):
        print("Processing image:", img_num, "/", len(image_paths))

    # Detect the largest face, then extract the face chip
    res = sdk.set_image(path)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set image at path: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # Detect the largest face in the image
    found, faceBoxAndLandmarks = sdk.detect_largest_face()
    if found == False:
        print(f"{Fore.RED}No face detected in image: {path}, not enrolling{Style.RESET_ALL}")
        continue

    # Get the aligned chip
    face = sdk.extract_aligned_face(faceBoxAndLandmarks)

    # Add the face chip to our array for processing
    face_chips.append(face)
    face_identities.append(os.path.basename(os.path.normpath(path)))

    # Only generate feature vector if we have enough face_chips as our specified batch size
    if len(face_chips) != batch_size:
        continue;

    # Generate feature vectors in batch
    generate_feature_vectors(face_chips, face_identities)


if len(face_chips) > 0:
    # We need to call generate feature vectors one final time to clear the array
    generate_feature_vectors(face_chips, face_identities)