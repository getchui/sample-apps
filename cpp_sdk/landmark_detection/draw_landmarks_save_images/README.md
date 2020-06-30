# Trueface SDK C++ Sample App
## Landmark Detection - Draw Landmarks and Save Images
This sample app demonstrates how to detect facial landmarks and bounding boxes using the SDK.
Images are loaded from a provided directory. The facial landmarks and face scores are drawn onto the images, then the images are saved to the run directory. 

# Demo
![alt text](./demo_images/family_landmarks.jpg)

![alt text](./demo_images/armstrong1_landmarks.jpg)

![alt text](./demo_images/obama1_landmarks.jpg)

![alt text](./demo_images/family2_landmarks.jpg)


### Prerequisites
Must have OpenCV installed.

### Build Instructions
* Place Trueface include files and `libtf.a` in `../trueface_sdk/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`
