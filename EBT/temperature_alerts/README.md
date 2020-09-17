# Temperature Based Alerts - Email and SMS Notifications

## Overview
The two sample apps in this directory demonstrate how to send alerts when the detected temperature exceeds a certain threshold.
The apps also use facial recognition to ensure we only send one notification per identity with elevated temperature.

## API Documentation
[Documentation](https://docs.google.com/document/d/1BAZO66pC694ZPZEqDvVaWI0cFafzIbz9qaNmkXRG0Tw/edit?usp=sharing)

[Postman Collection](https://documenter.getpostman.com/view/12009415/T17M7RNe?version=latest)

## Prerequisites - Common
- You will need to have the EBT application running on your camera.
- Download the python bindings library from [here](https://reference.trueface.ai/cpp/dev/latest/index.html#x86-64-python-bindings), then move the library to the same directory as `main.py`

## Prerequisites - Email
The `email.py` sample app uses the gmail smtp server. You will therefore need to have or create a gmail account to run the demo.
"Allow less secure apps" must be turned ON for the gmail account. This can be done at the following link: https://myaccount.google.com/lesssecureapps

## Prerequisites - SMS
- `pip install twilio`
- Create a Twilio account, purchase an SMS capable phone number, obtain an account sid and token.


## Running the Demo - Email
- Export your trueface token as an environment variable: `export TRUEFACE_TOKEN=${TRUEFACE_TOKEN}`
- Export the camera IP address as an environment variable: `export IP_ADDRESS=${CAMERA_IP_ADDRESS}`
- Replace the following lines with the sender email, recipient email, and password:
     
        # TODO: Enter the sender gmail account
        # "Allow less secure apps" must be turned ON https://myaccount.google.com/lesssecureapps
        self.sender_email = "sender_email@gmail.com"

        # TODO: Enter the recipient address
        self.receiver_email = "reciever_email@provier.com"

        # TODO: Enter your gmail account password (corresponding to the sender_email account)
        self.password = "password"

- Choose a desired threshold temperature:
        
        # TODO: Set this threshold to whatever you want
        self.temperature_threshold_C = 35 # In degrees C

- `python3 email.py`

## Running the Demo - SMS
- Export your trueface token as an environment variable: `export TRUEFACE_TOKEN=${TRUEFACE_TOKEN}`
- Export the camera IP address as an environment variable: `export IP_ADDRESS=${CAMERA_IP_ADDRESS}`
- Replace the following with your twilio account_sid, auth_token, and desired phone numbers:

        # Your Account Sid and Auth Token from twilio.com/console
        # DANGER! This is insecure. See http://twil.io/secure
        account_sid = 'ACe275517637075b7fb777f44f7b549efc'
        auth_token = 'your_auth_token'
        self.client = Client(account_sid, auth_token)

        self.from_number = '+15017122661'
        self.to_number = '+15558675310'


- Choose a desired threshold temperature:
        
        # TODO: Set this threshold to whatever you want
        self.temperature_threshold_C = 35 # In degrees C

- `python3 text.py`