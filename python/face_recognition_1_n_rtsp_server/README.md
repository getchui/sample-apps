# Trueface SDK Python Bindings Sample App
## Face Recognition RTSP Server
This sample app demonstrates how to consume an RTSP stream, decode the frames, run face recognition on those frames, then re-encode those frames as x264 and stream them over RTSP.

## Prerequisites
- Start by reading the `README.md` file [here](../README.md) for instructions on how to download the SDK and add the SDK to your environment.
Be sure to download the GPU enabled SDK for this sample app.

## Project Overview
- `enroll_in_database.py` - This script demonstrates how to use the Trueface SDK to enroll face recognition templates into a collection. 
This script must be run before running the main application. Open and modify the 
