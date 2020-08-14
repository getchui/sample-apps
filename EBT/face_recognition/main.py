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
                continue

            # Extract the feature vector
            res, v = self.sdk.get_largest_face_feature_vector()
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to generate feature vector")
                continue

            # Enroll the feature vector into the collection
            res, UUID = self.sdk.enroll_template(v, identity)
            if (res != tfsdk.ERRORCODE.NO_ERROR):
                print("Unable to enroll feature vector")
                continue

            # TODO: Can store the UUID for later use
            print("Success, enrolled template with UUID:", UUID)

    def on_message(ws, message):
        # data = json.loads(message)
        # print(data)
        url = 'http://192.168.0.12:8090/fr-template-lite'
        f = urllib.request.urlopen(url)
        print(f.read().decode('utf-8'))
        # print(data['mask_label'])

    def on_error(ws, error):
        print(error)

    def on_close(ws):
        print("---------------Connection Closed-------------")

    def on_open(ws):
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