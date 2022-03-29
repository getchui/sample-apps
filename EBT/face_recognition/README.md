# Face Recognition EBT Integration

## Overview
There are two sample apps contained within this project. Both demonstrate how to build a face recognition application which integrates with the EBT solution and how to consume the websocket stream.

The first app, named `fr_with_endpoints.py` achieves this by generating face recognition templates using the `GET /fr-template` or `GET /fr-template-lite` endpoints. These endpoints generate templates within the EBT application. In the case of the IRYX camera, these templates are generated directly onboard the camera. The IRYX camera currently only supports the LITE model because the full model has not yet been ported over (in progress). Therefore in order to use the FULL model with the IRYX EBT solution, follow the second solution, named `fr_with_sdk.py`. In this demo, the aligned face chips, which are sent as part of the websocket packet, are used to generate face recognition templates using the SDK.

Additionally, both sample apps demonstrate how to use the worker queue pattern. 


## API Documentation
[Documentation](https://docs.trueface.ai/elevated-body-temperature-access-control/ebt-installation-guide-getting-started-and-developer-api-v2)

[Postman Collection](https://docs.trueface.ai/elevated-body-temperature-access-control/ebt-postman-collection)

## Prerequisites
- You will need to have the EBT application running on your camera.
- Download the python bindings library from [here](https://reference.trueface.ai/cpp/dev/latest/index.html#x86-64-python-bindings), then move the library to this directory.

## Running the demo
- Export your trueface token as an environment variable: `export TRUEFACE_TOKEN=${TRUEFACE_TOKEN}`
- Export the camera IP address as an environment variable: `export IP_ADDRESS=${CAMERA_IP_ADDRESS}`
- `python3 fr_with_endpoints.py`
- `python3 fr_with_sdk.py`
