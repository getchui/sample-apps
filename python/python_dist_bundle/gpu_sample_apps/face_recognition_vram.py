# Sample code: Load images into GPU memory then generate face recognition templates

# This sample app demonstrates how to use the SDK with images loaded in GPU memory. 
# First, cupy is used to load images into GPU memory.
# Next, the SDK is used to run face detection on those GPU images, after which the face chips are extracted.
# Finally, we generate face recognition templates for those face chips using batching, then compute the similarity score. 


import os
import tfsdk
from colorama import Fore
from colorama import Style

options = tfsdk.ConfigurationOptions()

# Enable the GPU SDK
options.GPU_options = True  # Batching is only supported by GPU

options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.TFV5
# Note, if you do use TFV5, you will need to run the download script in /download_models to obtain the model file

sdk = tfsdk.SDK(options)

# TODO: export your license token as TRUEFACE_TOKEN environment variable
is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
if (is_valid == False):
    print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
    print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
    quit()

import cv2
import numpy as np
img_1 = cv2.imread('../images/brad_pitt_1.jpg')
img_2 = cv2.imread('../images/brad_pitt_2.jpg')

import cupy as cp

# load the image into the graphics card's memory
img_gpu_1 = cp.asarray(img_1)

# pass the gpu memory address to the sdk
res = sdk.set_image(img_gpu_1.data.ptr, img_1.shape[1], img_1.shape[0], tfsdk.COLORCODE.bgr, 0)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 1{Style.RESET_ALL}")
    quit()

# detect the faces in the image without copying the image into the cpu memory
faces_1 = sdk.detect_faces()
if not faces_1:
    print(f"{Fore.RED}Unable to detect any faces in image 1!{Style.RESET_ALL}")
    quit()

# create an array to hold the face chip in the gpu memory
face_chip_1 = cp.zeros((112,112,3), 'uint8')

# the sdk copies the face chip into the allocated memory
sdk.extract_aligned_face(face_chip_1.data.ptr, faces_1[0])

# load the image into the graphics card's memory
img_gpu_2 = cp.asarray(img_2)

# pass the gpu memory address to the sdk
res = sdk.set_image(img_gpu_2.data.ptr, img_2.shape[1], img_2.shape[0], tfsdk.COLORCODE.bgr, 0)
if (res != tfsdk.ERRORCODE.NO_ERROR):
    print(f"{Fore.RED}Unable to set image 2{Style.RESET_ALL}")
    quit()

# detect the faces in the image without copying the image into the cpu memory
faces_2 = sdk.detect_faces()
if not faces_1:
    print(f"{Fore.RED}Unable to detect any faces in image 2!{Style.RESET_ALL}")
    quit()

face_chip_2 = cp.zeros((112,112,3), 'uint8')
sdk.extract_aligned_face(face_chip_2.data.ptr, faces_2[0])


# check the face chip to see it looks fine
chip_img_1_cpu = face_chip_1.get()
cv2.imwrite('/tmp/face_1.jpg', cv2.cvtColor(chip_img_1_cpu, cv2.COLOR_BGR2RGB))

chip_img_2_cpu = face_chip_2.get()
cv2.imwrite('/tmp/face_2.jpg', cv2.cvtColor(chip_img_2_cpu, cv2.COLOR_BGR2RGB))

# extract faceprints using batching
face_chips = [face_chip_1.data.ptr, face_chip_2.data.ptr]
status, faceprints = sdk.get_face_feature_vectors(face_chips)

# compare the two faceprints, this should return a high match probability
res = faceprints[0].compare(faceprints[1])
print(res)

