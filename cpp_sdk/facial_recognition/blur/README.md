# Trueface SDK C++ Sample App
## Facial Recognition - 1:N Identification with Blur
This sample app demonstrates the use of 1:N identification with blur.
Face Recognition templates are first generated and enrolled into a collection with their associated ID.
The quality of these enrollment images are checked before enrolling into the collection as we want to ensure only high quality images are enrolled in collections.
Meanwhile, a separate thread is launched which grabs frames from an RTSP stream.
Face recognition is then run on these RTSP frames. 
If any of the identities match those in the collection, the identity and a green bounding box is draw on the video stream.
If the identity is not recognized, then the face is blurred and a white bounding box is drawn on the face.
The video stream is then displayed in real time.  

### Demo
![alt text](./demo_gifs/demo1.gif)

![alt text](./demo_gifs/demo2.gif)

![alt text](./demo_gifs/demo3.gif)

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface include files and `libtf.a` in `../../trueface_sdk/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`
