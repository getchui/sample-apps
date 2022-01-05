# Sample code: Generate face recognition templates using GPU batching.

# This sample app demonstrates how to use batching with the GPU SDK. 
# First, we extract the face chip for several images
# Next, we generate face recognition templates in batch.
# Batching increases the GPU throughput. 
# Finally, we generate the similarity scores.

import tfsdk
import os
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()

# Enable the GPU SDK
options.GPU_options = True  # Batching is only supported by GPU

# In this sample app we will be using a batch size of 3
batch_size = 3

options.GPU_options.face_recognizer_GPU_options.opt_batch_size = batch_size
options.GPU_options.face_recognizer_GPU_options.max_batch_size = batch_size



options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

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



# List of images to use for our batch template generation
images = [
    "../images/brad_pitt_1.jpg",
    "../images/brad_pitt_2.jpg", 
    "../images/brad_pitt_3.jpg"
]

# Create array to store our face chips
face_chips = []

# First need to run face detection and generate face chips sequentially.
for image in images:
    res = sdk.set_image(image)
    if (res != tfsdk.ERRORCODE.NO_ERROR):
        print(f"{Fore.RED}Unable to set image: {image}{Style.RESET_ALL}")
        quit()


     # Detect the largest face in the image
    found, face_bounding_box = sdk.detect_largest_face()

    if not found:
        print(f"{Fore.RED}Could not find face in image!{Style.RESET_ALL}")
        quit()
        

    face_chip = sdk.extract_aligned_face(face_bounding_box)
    face_chips.append(face_chip)    


# Now that we have generated the face chips, we can go ahead and batch generate the FR templates.
res, faceprints = sdk.get_face_feature_vectors(face_chips)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to extract face feature vectors{Style.RESET_ALL}")
    quit()

# Run similarity comparisons
res, match_prob, sim_score = sdk.get_similarity(faceprints[0], faceprints[1])
print(f"Image 1 vs Image 2 match probability: {match_prob}")

res, match_prob, sim_score = sdk.get_similarity(faceprints[1], faceprints[2])
print(f"Image 2 vs Image 3 match probability: {match_prob}")

