# Trueface Temperature Detection Frontend Sample App

## Overview
This sample app demonstrates how to build a simple frontend app which connects to the Trueface Temperature Detection and Access Control application.
To learn more, please visit: https://docs.trueface.ai/access-control 

## API Documentation
[Documentation](https://docs.google.com/document/d/1BAZO66pC694ZPZEqDvVaWI0cFafzIbz9qaNmkXRG0Tw/edit?usp=sharing)

[Postman Collection](https://documenter.getpostman.com/view/12009415/T17M7RNe?version=latest)



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

## Docker
Instead of building the project manually, you can run the docker container instead.

`docker run -p 5000:5000 -d --name tf-ebt-frontend trueface/ebt-frontend`

Next, go to `http://0.0.0.0:5000` in your browser.

Note, if on windows, you may have to go to `http://127.0.0.1:5000`

### Demo
![alt text](./demo/demo.gif)