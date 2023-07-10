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
* Place Trueface header files in `../../trueface_sdk/include/` and trueface libraries in `../../trueface_sdk/lib/`
* Download the appropriate face recognition model using the download scripts which come bundled with the SDK. 
* Export your Trueface token to your environment as `TRUEFACE_TOKEN`.
  Alternatively, open the `CMakeLists.txt` file and edit this line here: `add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")`.
  Replace `YOUR_TOKEN_HERE` with the license token you were provided with. If you have not yet received a token, contact support@pangiam.com.
* `mkdir build && cd build`
* `cmake ..`
* `make`
