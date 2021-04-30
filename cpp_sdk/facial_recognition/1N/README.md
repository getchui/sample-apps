# Trueface SDK C++ Sample App
## Facial Recognition - 1:N Identification
This sample app demonstrates the use of 1:N matching. 
Face Recognition templates are first generated and enrolled into a collection with their associated ID.
The quality of these enrollment images are checked before enrolling into the collection as we want to ensure only high quality images are enrolled in collections.
Meanwhile, a separate thread is launched which grabs frames from an RTSP stream.
Face recognition is then run on these RTSP frames. 
If any of the identities match those in the collection, the identity and bounding box is draw on the video stream.
The video stream is then displayed in real time.  

### Demo
![alt text](./demo_gifs/demo1.gif)

![alt text](./demo_gifs/demo2.gif)

![alt text](./demo_gifs/demo3.gif)

![alt text](./demo_gifs/demo4.gif)

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface include files and `libtf.a` in `../../trueface_sdk/`
* Download the appropriate face recognition model using the download scripts which come bundled with the SDK. 
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`
