# Trueface SDK Python Bindings Sample App
## Active Spoof Frontend App
This sample app demonstrates how to run the active spoof check pipeline using your web camera.


## Prerequisites
- Start by reading the `README.md` file [here](../README.md) for instructions on how to download the SDK and add the SDK to your environment.
Using the GPU enabled SDK will improve inference speed.
- `pip3 install opencv-python`
- `pip3 install colorama`
- `pip3 install pynput`

## Project Overview
- `active_spoof.py` demonstrates the steps necessary to run active spoof detection and prevention. 
  Active spoof works by analyzing the way a persons face changes as they move closer to a camera. 
  The active spoof solution therefore required two images and expects the face a certain distance from the camera. 
  In the far image, the face should be about 18 inches from the camera, while in the near image, the face should be 7-8 inches from the camera.
- The sample app works by tracking state and requires the user to first capture a far image, then to move their face closer to the camera and capture a near shot image.
Once both images have been captured, active spoof detection is run and the results are displayed on the screen.
- By advised, the first time you capture an image it will appear to freeze briefly. This is expected and is due to lazy initialization of the SDK (when it appears to freeze, the face recognition model is being decrypted and read from disk into memory).

## Demo
The following demo shows the sample app being run with a real image, and also with a spoof attempt image.




## Running the demo
- `python3 active_spoof.py`
- Follow the instructions drawn on the video.
