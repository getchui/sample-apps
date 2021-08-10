import websocket
import json
import urllib.request
import urllib.parse
import os
from multiprocessing import Process, Queue
from multiprocessing import set_start_method
import time
import numpy as np
import cv2
import base64

try:
    import thread
except ImportError:
    import _thread as thread

import tfsdk


class Controller:
    def __init__(self, token, ip):
        self.packetCount = 0

        self.ip = ip

        # Choose our configuration options here to ensure all child processes initialize the SDK using the same options
        # We will be passing the configuration options to the child processes which utilize the SDK
        # Since we will be using the GET /fr-template-lite endpoint, we will initialize the SDK using the lite model
        options = tfsdk.ConfigurationOptions()
        options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.FULL # Use the full model
        options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.NONE # We will NOT be writing template to disk for this demo.
        options.smallest_face_height = -1 # https://reference.trueface.ai/cpp/dev/latest/py/general.html#tfsdk.ConfigurationOptions

        # Launch a worker process to generate templates from the aligned face chips, and another process to run identification with those templates
        self.faceChipQueue = Queue()
        self.templateQueue = Queue()

        identifyProcess = Process(target=self.identifyTemplate, args=(self.templateQueue, token, options))
        identifyProcess.daemon = True
        identifyProcess.start()

        templateGenerationProcess = Process(target=self.generateTemplate, args=(self.templateQueue, self.faceChipQueue, token, options))
        templateGenerationProcess.daemon = True
        templateGenerationProcess.start()

        # Create processes to monitor the queue size
        templateQueueStatusProcess = Process(target=self.printQueueStatus, args=(self.templateQueue, "Template Queue"))
        templateQueueStatusProcess.daemon = True
        templateQueueStatusProcess.start()

        facechipQueueStatusProcess = Process(target=self.printQueueStatus, args=(self.faceChipQueue, "Facechip Queue"))
        facechipQueueStatusProcess.daemon = True
        facechipQueueStatusProcess.start()

        self.start_connection()

    def on_message(self, message):
        self.packetCount = self.packetCount + 1

        # First, parse the response
        response = json.loads(message)

        # TODO: Do something with the temp, mask status, etc. 
        # ex: can prompt the user to put on a mask, remove mask for FR, remove glasses for temp, etc
        # print("Avg Temp:", response['average_temperature_measured'], response['temp_unit'])

        # Mask label only set when there is a face in the frame
        # if 'mask_label' in response.keys():
            # print("Mask Status:", response["mask_label"])

        # Check if there is a face in the frame
        if response["face_detected"] == True:
            # Only process every 2th packet, otherwise our system can get overloaded since the packets are coming in faster than the CPU can generate FULL model templates
            # If running the SDK on GPU, this is probably not an issue as we can generate packets much faster
            if self.packetCount % 2 != 0:
                return

            self.packetCount = 0

            # Add the aligned face chip to the face chip queue to be processed by our worker process
            self.faceChipQueue.put(response['face_chip_aligned'])

    def on_error(self, error):
        print("---------------Error-------------------------")
        print(error)

    def on_close(self):
        print("---------------Connection Closed-------------")

    def on_open(self):
        print("---------------Connection Opened-------------")


    def start_connection(self):
        # Creates the websocket connection
        websocket.enableTrace(False)
        websocket_ip = "ws://" + self.ip + ":8091"
        print(websocket_ip)
        self.ws = websocket.WebSocketApp(websocket_ip,
                              on_message = self.on_message,
                              on_error = self.on_error,
                              on_close = self.on_close,
                              on_open = self.on_open)
        self.ws.run_forever()

    def printQueueStatus(self, queue, queuename):
        while True:
            # Sleep for 1 second, then print the size of the queue
            time.sleep(1)
            print("[STATUS]", queuename, "size:", queue.qsize())


    def generateTemplate(self, templateQueue, faceChipQueue, token, options):
        # Initialize the trueface SDK
        sdk = tfsdk.SDK(options)

        # Set the SDK license
        is_valid = sdk.set_license(token)
        if (is_valid == False):
            print("Invalid License Provided")
            quit() 


        # When a new face chip image becomes available, generate a template for that face chip
        while True:
            alignedFaceChip = faceChipQueue.get()
            
            # Decode the base64 encoded face chip
            binary = base64.b64decode(alignedFaceChip)
            image = np.asarray(bytearray(binary), dtype="uint8")
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            res, faceprint = sdk.get_face_feature_vector(frame)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to generate feature vector.")
                continue

            # Now push the template to the template queue to be processed by the other worker process
            templateQueue.put(faceprint)


    def identifyTemplate(self, templateQueue, token, options):
        # Initialize the trueface SDK
        sdk = tfsdk.SDK(options)

        # Set the SDK license
        is_valid = sdk.set_license(token)
        if (is_valid == False):
            print("Invalid License Provided")
            quit() 


        # At this point, we can either load an existing trueface database & collection, or create a new one and populate it with data
        # For the sake of this demo, we will create a new collection (memory only, meaning data will not persist after app shuts down) 
        
        # If we are using a database backend, need to first call create_database_connection
        # Since we are using DATABASEMANAGEMENTSYSTEM.NONE this is not necessary
        # For more aobut 1 to N, please refer to: https://reference.trueface.ai/cpp/dev/latest/usage/identification.html

        # Create a new collection
        res = sdk.create_load_collection("my_collection")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Unable to create collection")
            quit()

        # Some identities to populate into our collection
        # TODO: Can add images of yourself here
        image_identities = [
            ("../../images/armstrong/armstrong1.jpg", "Armstrong"),
            ("../../images/armstrong/armstrong2.jpg", "Armstrong"), # Can add the same identity more than once
            ("../../images/obama/obama1.jpg", "Obama")
        ]

        # Generate templates, enroll in our collection
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
            # Therefore, ensure that the face height is at least 100px
            faceHeight = faceBoxAndLandmarks.bottom_right.y - faceBoxAndLandmarks.top_left.y
            print("Face height:", faceHeight, "pixels")

            if faceHeight < 100:
                print("The face is too small in the image for a high quality enrollment.")
                continue

            # Get the aligned chip so we can compute the image quality
            face = sdk.extract_aligned_face(faceBoxAndLandmarks)

            # Compute the image quality score
            res, quality = sdk.estimate_face_image_quality(face)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("There was an error computing the image quality")
                continue

            # Ensure the image quality is above a threshold (TODO: adjust this threshold based on your use case).
            print("Face quality:", quality)
            if quality < 0.8:
                print("The image quality is too poor for enrollment")
                continue

            # As a final check, we can check the orientation of the head and ensure that it is facing forward
            # To see the effect of yaw and pitch on match score, refer to: https://reference.trueface.ai/cpp/dev/latest/py/face.html#tfsdk.SDK.estimate_head_orientation

            res, yaw, pitch, roll = sdk.estimate_head_orientation(faceBoxAndLandmarks)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to compute head orientation")

            print("Head orientation, Yaw:", yaw * 180 / 3.14, ", Pitch:", pitch * 180 / 3.14, ", Roll:", roll * 180 / 3.14, "degrees")
            # TODO: Can filter out images with extreme yaw and pitch here

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

        # When a new template becomes avaliable, run the identification function
        while True:
            probe_faceprint = templateQueue.get()

            # Now run 1 to N identification
            ret_code, found, candidate = sdk.identify_top_candidate(probe_faceprint)
            if ret_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Something went wrong!")
            elif found == True:
                print("Found match with identity:", candidate.identity)
                print("Match probability:", candidate.match_probability)
            else:
                print("Unable to find match")
            print(" ")



if __name__ == '__main__':
    set_start_method("spawn")

    ip_address = os.environ['IP_ADDRESS']
    trueface_token = os.environ['TRUEFACE_TOKEN']

    controller = Controller(trueface_token, ip_address)