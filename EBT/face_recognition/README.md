# Face Recognition EBT Integration

## Overview
This sample app demonstrates how to build a facial recognition application and integrate with the EBT solution

## API Documentation
[Documentation](https://docs.google.com/document/d/1BAZO66pC694ZPZEqDvVaWI0cFafzIbz9qaNmkXRG0Tw/edit?usp=sharing)

[Postman Collection](https://documenter.getpostman.com/view/12009415/T17M7RNe?version=latest)

## Prerequisites
- You will need to have the EBT application running on your camera.
- Download the python bindings library from [here](https://reference.trueface.ai/cpp/dev/latest/index.html#x86-64-python-bindings), then move the library to the same directory as `main.py`

## Running the demo
- Export your trueface token as an environment variable: `export TRUEFACE_TOKEN=${TRUEFACE_TOKEN}`
- Export the camera IP address as an environment variable: `export IP_ADDRESS=${CAMERA_IP_ADDRESS}`
- `python3 main.py`
