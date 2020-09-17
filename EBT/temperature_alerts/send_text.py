import websocket
import json
import urllib.request
import urllib.parse
import os
from datetime import datetime

# Download the helper library from https://www.twilio.com/docs/python/install
from twilio.rest import Client

try:
    import thread
except ImportError:
    import _thread as thread

import tfsdk


class ConnectionHandler:
    def __init__(self, token, ip):
        self.ip = ip

        # TODO: Set this threshold to whatever you want
        self.temperature_threshold_C = 38.5 # In degrees C

        # Initialize the trueface SDK using hte lite model
        # We will be using facial recognition to ensure we only send 1 notification per identity
        # So that if a sick person stands infront of the camera, we only send 1 notification and not countless
        options = tfsdk.ConfigurationOptions()
        options.fr_model = tfsdk.FACIALRECOGNITIONMODEL.LITE
        self.sdk = tfsdk.SDK(options)

        is_valid = self.sdk.set_license(token)
        if (is_valid == False):
            print("Invalid License Provided")
            quit() 

        self.faceprint = {}

        # Set up text message stuff

        # Your Account Sid and Auth Token from twilio.com/console
        # DANGER! This is insecure. See http://twil.io/secure
        account_sid = 'ACe275517637075b7fb777f44f7b549efc'
        auth_token = 'your_auth_token'
        self.client = Client(account_sid, auth_token)

        self.from_number = '+15017122661'
        self.to_number = '+15558675310'


    def send_email(self, temperature):
        # Get timestamp
        now = datetime.now()
        dt_string = now.strftime("%B %d, %Y %H:%M:%S")

        # Populate the email with some information
        temp_str = "{:.2f}".format(temperature)
        msg_body = "Alert! Elevated body temperature of " + temp_str + "°C detected at " + dt_string + "."

        message = self.client.messages \
            .create(
                 body=msg_body,
                 from_=self.from_number,
                 to=self.to_number
             )

        print(service.sid)


    def on_message(self, message):
        # This is where we do the bulk of the processing

        # First, parse the response
        response = json.loads(message)

        # Next, determine if the termperature was detected
        if response['eyeduct_visible'] == True:

            avg_temp_C = response['average_temperature_measured']

            if (response['temp_unit'] == 'F'):
                avg_temp_C = (avg_temp_C - 32) * 5 / 9

            # Next, check if the temperature exceeds the threshold temperture
            if (avg_temp_C > self.temperature_threshold_C):

                # Generate a face recognition template
                endpoint = "http://" + self.ip + ":8090/fr-template-lite"
                f = urllib.request.urlopen(endpoint)
                decoded = json.loads(f.read().decode('utf-8'))

                if 'error' in decoded.keys():
                    print("There was an error getting the template")
                    return;

                else:
                    # Create a faceprint, populate it
                    errorcode, probe_faceprint = tfsdk.SDK.json_to_faceprint(json.dumps(decoded))
                    if (errorcode != tfsdk.ERRORCODE.NO_ERROR):
                        print("There was an error decoding the json to a faceprint")
                        return

                    # Check if we previously detected a face with elevated temperature
                    if self.faceprint:
                        # Check to see if the identity is the same
                        res =  self.faceprint.compare(probe_faceprint)
                        if res["cosine_similarity"] > 0.2:
                            # Same identity, so let's skip. 
                            # We already sent a notification for this identity
                            return

                    # Update the current identity in the frame
                    self.faceprint = probe_faceprint

                    # Send a notification
                    self.send_email(avg_temp_C)

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


ip_address = os.environ['IP_ADDRESS']
trueface_token = os.environ['TRUEFACE_TOKEN']

connectionHandler = ConnectionHandler(trueface_token, ip_address)
connectionHandler.start_connection()