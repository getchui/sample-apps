# Trueface Temperature Detection Frontend Sample App

## Overview
This sample app demonstrates how to build a simple frontend app which connects to the Trueface Temperature Detection and Access Control application.
To learn more, please visit: https://docs.trueface.ai/elevated-body-temperature-access-control 

## API Documentation
[Documentation](https://docs.trueface.ai/elevated-body-temperature-access-control/ebt-installation-guide-getting-started-and-developer-api-v2)

[Postman Collection](https://docs.trueface.ai/elevated-body-temperature-access-control/ebt-postman-collection)



## Building the demo
To build the demo, run the following commands:

`cd frontend`

`npm install`

`npm run-script build`

## Running the demo
To run the demo, run the following command:

`python3 main.py`

Next, go to `http://0.0.0.0:5000` in your browser.

Note, if on windows, you may have to go to `http://127.0.0.1:5000`

If connecting to the IRYX camera, use the IP address of the camera.
If connecting to the FLIR A400 camera, use the IP address of the device running the Trueface software.


Important: If using an IRYX camera, you must ensure the streams are encoded as x264. This can be done from the web interface, using the top navigation bar, click on *Visible* > *Video Settings* then change the encoding settings to x264. You will then need to reboot the camera from the home window.

## Docker
Instead of building the project manually, you can run the docker container instead.

`docker run -p 5000:5000 -d --name tf-ebt-frontend trueface/ebt-frontend`

Next, go to `http://0.0.0.0:5000` in your browser.

Note, if on windows, you may have to go to `http://127.0.0.1:5000`

### Demo
![alt text](./demo/demo.gif)
