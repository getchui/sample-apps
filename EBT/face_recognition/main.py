import websocket
import json
import urllib.request
import urllib.parse
import os
try:
    import thread
except ImportError:
    import _thread as thread

import tfsdk


class ConnectionHandler:
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

        # At this point, we can either load an existing database & collection, or create a new one and populate it with data
        # For the sake of this demo, we will create a new collection (memory only, meaning data will not persist after app shuts down) 
        
        # If we are using a database backend, need to first call create_database_connection
        # Since we are using DATABASEMANAGEMENTSYSTEM.NONE this is not necessary

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
            # Generate a template for each image
            res = self.sdk.set_image(path)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to set image at path:", path)
                quit()

            # Extract the feature vector
            res, v = self.sdk.get_largest_face_feature_vector()
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to generate feature vector")
                quit()

            # Enroll the feature vector into the collection
            res, UUID = self.sdk.enroll_template(v, identity)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to enroll feature vector")
                quit()

            # TODO: Can store the UUID for later use
            print("Success, enrolled template with UUID:", UUID)

    def on_message(self, message):
        # This is where we do the bulk of the processing

        # First, parse the response
        response = json.loads(message)

        # TODO: Do something with the temp, mask status, etc. 
        print("Avg Temp:", response['average_temperature_measured'], response['temp_unit'])

        # Mask label only set when there is a face in the frame
        if 'mask_label' in response.keys():
            print("Mask Status:", response["mask_label"])

        # Check if there is a face in the frame
        if response["face_detected"] == True:
            # If there is a face, we can request a template
            endpoint = "http://" + self.ip + ":8090/fr-template-lite"
            f = urllib.request.urlopen(endpoint)
            decoded = json.loads(f.read().decode('utf-8'))
            
            # Create a faceprint, populate it
            probe_faceprint = tfsdk.Faceprint()
            probe_faceprint.model_name = decoded["model_name"]
            probe_faceprint.model_options = decoded["model_options"]
            probe_faceprint.sdk_version = decoded["sdk_version"]
            probe_faceprint.feature_vector = decoded["feature_vector"]

            # Now run 1 to N identification
            ret_code, found, candidate = self.sdk.identify_top_candidate(probe_faceprint)
            if ret_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Something went wrong!")
            elif found == True:
                print("Found match with identity:", candidate.identity)
                print("Match probability:", candidate.match_probability)
            else:
                print("Unable to find match")

        print()
            


    def on_error(self, error):
        print("---------------Error-------------------------")
        print(error)

    def on_close(self):
        print("---------------Connection Closed-------------")

    def on_open(self):
        print("---------------Connection Opened-------------")
        

    def start_connection(self):
        # Creates the websocket connection
        websocket.enableTrace(True)
        websocket_ip = "ws://" + self.ip + ":8091"
        print(websocket_ip)
        self.ws = websocket.WebSocketApp(websocket_ip,
                              on_message = self.on_message,
                              on_error = self.on_error,
                              on_close = self.on_close,
                              on_open = self.on_open)
        self.ws.run_forever()


ip_address = os.environ['IP_ADDRESS']
trueface_token = os.environ['TRUEFACE_TOKEN']

connectionHandler = ConnectionHandler(trueface_token, ip_address)
connectionHandler.start_connection()