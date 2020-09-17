# Temperature Based Email Alerts

## Overview
This sample app demonstrates how to send email alerts when the detected temperature exceeds a certain threshold.
The app also uses facial recognition to ensure we only send one notification per identity with elevated temperature.

## API Documentation
[Documentation](https://docs.google.com/document/d/1BAZO66pC694ZPZEqDvVaWI0cFafzIbz9qaNmkXRG0Tw/edit?usp=sharing)

[Postman Collection](https://documenter.getpostman.com/view/12009415/T17M7RNe?version=latest)

## Prerequisites
- You will need to have the EBT application running on your camera.
- Download the python bindings library from [here](https://reference.trueface.ai/cpp/dev/latest/index.html#x86-64-python-bindings), then move the library to the same directory as `main.py`

## Running the demo
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

- `python3 main.py`
