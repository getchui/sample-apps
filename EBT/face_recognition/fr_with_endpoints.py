import websocket
import json
import urllib.request
import urllib.parse
import os
from multiprocessing import Process, Queue
import time

try:
    import thread
except ImportError:
    import _thread as thread

import tfsdk


class Controller:
    def __init__(self, token, ip):
        self.ip = ip

        # Initialize the trueface SDK
        # Be sure to use the same SDK options as the camera
        # Since we will be using the GET /fr-template-lite endpoint, we will initialize the SDK using the lite model
        options = tfsdk.ConfigurationOptions()
        options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.LITE
        options.dbms = tfsdk.DATABASEMANAGEMENTSYSTEM.NONE # We will NOT be writing template to disk for this demo.
        options.smallest_face_height = -1 # https://reference.trueface.ai/cpp/dev/latest/py/general.html#tfsdk.ConfigurationOptions
        self.sdk = tfsdk.SDK(options)

        is_valid = self.sdk.set_license(token)
        if (is_valid == False):
            print("Invalid License Provided")
            quit() 


        # At this point, we can either load an existing trueface database & collection, or create a new one and populate it with data
        # For the sake of this demo, we will create a new collection (memory only, meaning data will not persist after app shuts down) 
        
        # If we are using a database backend, need to first call create_database_connection
        # Since we are using DATABASEMANAGEMENTSYSTEM.NONE this is not necessary
        # For more aobut 1 to N, please refer to: https://reference.trueface.ai/cpp/dev/latest/usage/identification.html

        # Create a new collection
        res = self.sdk.create_load_collection("my_collection")
        if (res != tfsdk.ERRORCODE.NO_ERROR):
            print("Unable to create collection")
            quit()

        # Some identities to populate into our collection
        # TODO: Can add images of yourself here
        image_identities = [
            ("../../cpp_sdk/images/armstrong/armstrong1.jpg", "Armstrong"),
            ("../../cpp_sdk/images/armstrong/armstrong2.jpg", "Armstrong"), # Can add the same identity more than once
            ("../../cpp_sdk/images/obama/obama1.jpg", "Obama")
        ]

        # Generate templates, enroll in our collection
        for path, identity in image_identities:
            print("Processing image:", path, "with identity:", identity)
            # Generate a template for each image
            res = self.sdk.set_image(path)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to set image at path:", path)
                continue

            # Detect the largest face in the image
            found, faceBoxAndLandmarks = self.sdk.detect_largest_face()
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
            face = self.sdk.extract_aligned_face(faceBoxAndLandmarks)

            # Compute the image quality score
            res, quality = self.sdk.estimate_face_image_quality(face)
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

            res, yaw, pitch, roll = self.sdk.estimate_head_orientation(faceBoxAndLandmarks)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to compute head orientation")

            print("Head orientation, Yaw:", yaw * 180 / 3.14, ", Pitch:", pitch * 180 / 3.14, ", Roll:", roll * 180 / 3.14, "degrees")
            # TODO: Can filter out images with extreme yaw and pitch here

            # Now that we have confirmed the images are high quality, generate a template from that image
            res, faceprint = self.sdk.get_face_feature_vector(face)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("There was an error generating the faceprint")
                continue

            # Enroll the feature vector into the collection
            res, UUID = self.sdk.enroll_template(faceprint, identity)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to enroll feature vector")
                continue

            # TODO: Can store the UUID for later use
            print("Success, enrolled template with UUID:", UUID)
            print("--------------------------------------------")

        # Launch a new process to process all the incomming templates
        self.templateQueue = Queue()

        self.identifyProcess = Process(target=self.identifyTemplate, args=(self.templateQueue, options, token))
        self.identifyProcess.daemon = True
        self.identifyProcess.start()

        self.start_connection()

    def on_message(self, message):
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
            # If there is a face, we can request a template
            # TODO: You can choose to only request a template every nth frame with a face
            endpoint = "http://" + self.ip + ":8090/fr-template-lite"
            f = urllib.request.urlopen(endpoint)
            decoded = json.loads(f.read().decode('utf-8'))

            if 'error' in decoded.keys():
                print("There was an error getting the template")
            else:
                # Add the template to the queue to be processed
                self.templateQueue.put(json.dumps(decoded))

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

    def identifyTemplate(self, templateQueue, sdkOptions, token):
        # Create another instance of the SDK with the same configuration options
        sdk = tfsdk.SDK(sdkOptions)

        is_valid = sdk.set_license(token)
        if (is_valid == False):
            print("Invalid License Provided")
            quit() 

        while True:
            template = templateQueue.get()
            errorcode, probe_faceprint = tfsdk.SDK.json_to_faceprint(template)
            if (errorcode != tfsdk.ERRORCODE.NO_ERROR):
                print("There was an error decoding the json to a faceprint")
                continue
            # Now run 1 to N identification
            ret_code, found, candidate = sdk.identify_top_candidate(probe_faceprint)
            if ret_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Something went wrong!")
            elif found == True:
                print("Found match with identity:", candidate.identity)
                print("Match probability:", candidate.match_probability)
            else:
                print("Unable to find match")



if __name__ == '__main__':
    ip_address = os.environ['IP_ADDRESS']
    trueface_token = os.environ['TRUEFACE_TOKEN']

    controller = Controller(trueface_token, ip_address)