# Sample code: Load collection from disk then run 1N identification

# This sample app demonstrates how to use 1N identification. First, an existing collection (created by running enroll_in_database) is loaded from disk.
# Next, 1N identification is run to determine the identity of an anonymous template.
# Note, you must run enroll_in_database before being able to run this sample app. 

import tfsdk
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()
# Can set configuration options here

# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.NONE # The data will not persist after the application terminates

# Load the collection from an SQLITE database
options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.SQLITE

# If you previously enrolled the templates into a PostgreSQL database, then use POSTGRESQL instead
# options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.POSTGRESQL

# To enable database encryption...
# encryptDatabase = tfsdk.EncryptDatabase()
# encryptDatabase.enable_encryption = True
# encryptDatabase.key = "TODO: Your encryption key here"
# options.encrypt_database = encryptDatabase


options.smallest_face_height = 40 # https://reference.trueface.ai/cpp/dev/latest/usage/general.html#_CPPv4N8Trueface20ConfigurationOptions18smallestFaceHeightE
options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5 
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

# TODO: If you have a NVIDIA gpu, then enable the enableGPU flag (you will require a GPU specific token for this).
# options.GPU_options = True

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

# Connect to our existing database
res = sdk.create_database_connection("my_database.db")
if (res != tfsdk.ERRORCODE.NO_ERROR):
  print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
  quit()


# ex. If using POSTGRESQL backend...
# res = sdk.create_database_connection("host=localhost port=5432 dbname=my_database user=postgres password=admin")
# if (res != tfsdk.ERRORCODE.NO_ERROR):
  # print(f"{Fore.RED}Unable to create database connection{Style.RESET_ALL}")
  # quit()

  
# Load the existing collection into memory
res = sdk.create_load_collection("my_collection")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to load collection{Style.RESET_ALL}")
    quit()

# Use a new image of Brad Pitt as the probe
res = sdk.set_image("../images/brad_pitt_1.jpg")
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image{Style.RESET_ALL}")
    quit()

# Extract the feature vector
# We do not need to check the quality of the probe face templates
# We mainly want to ensure that the enrollment templates are high quality, 
# This is less of a concern with probe templates 
res, v1, found_face = sdk.get_largest_face_feature_vector()
if (res != tfsdk.ERRORCODE.NO_ERROR or found_face != True):
    print(f"{Fore.RED}No face found in image, unable to generate feature vector{Style.RESET_ALL}")
    quit()

# Run identification query using threshold of 0.4
res, found, candidate = sdk.identify_top_candidate(v1, 0.4)

if found == False or res != tfsdk.ERRORCODE.NO_ERROR:
    print(f"{Fore.RED}Unable to find match above threshold{Style.RESET_ALL}")
else:
    print("Match found!")
    print(f"Identity: {candidate.identity}")
    print(f"Match Probability: {candidate.match_probability}")
    print(f"Similarity: {candidate.similarity_measure}")