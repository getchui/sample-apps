# Trueface Temperature Detection Frontend Sample App

## Overview
This sample app demonstrates how to build a simple frontend app which connects to the Trueface Temperature Detection and Access Controll application.
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
To run the following demo, run the following command:
`python3 main.py`

Next, go to `http://0.0.0.0:5000` in your browser.

If connecting to the IRYX camera, use the IP address of the camera.
If connecting to the FLIR A400 camera, use the IP address of the device running the Trueface software.

### Demo
![alt text](./demo/demo.gif)
