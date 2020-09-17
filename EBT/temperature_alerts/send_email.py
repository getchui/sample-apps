import websocket
import json
import urllib.request
import urllib.parse
import os
import smtplib, ssl
from datetime import datetime
from email.message import EmailMessage

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

        # Set up email stuff
        # For this demo, we will be using the gmail smtp server 
        self.port = 465  # For SSL
        self.smtp_server = "smtp.gmail.com"

        # TODO: Enter the sender gmail account
        # "Allow less secure apps" must be turned ON https://myaccount.google.com/lesssecureapps
        self.sender_email = "sender_email@gmail.com"

        # TODO: Enter the recipient address
        self.receiver_email = "reciever_email@provier.com"

        # TODO: Enter your gmail account password (corresponding to the sender_email account)
        self.password = "password"


    def send_email(self, temperature):
        # Get timestamp
        now = datetime.now()
        dt_string = now.strftime("%B %d, %Y %H:%M:%S")

        # Populate the email with some information
        msg = EmailMessage()
        temp_str = "{:.2f}".format(temperature)
        msg_body = "Elevated body temperature of " + temp_str + "°C detected at " + dt_string + "."
        msg.set_content(msg_body)

        msg['Subject'] = "Alert: Elevated body temperature detected"
        msg['From'] = self.sender_email
        msg['To'] = self.receiver_email

        # Send the email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.smtp_server, self.port, context=context) as server:
            server.login(self.sender_email, self.password)
            server.send_message(msg)
            print("Notification sent!")


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