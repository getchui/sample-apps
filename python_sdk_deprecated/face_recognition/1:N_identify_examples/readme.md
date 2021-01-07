# Face Recognition Sample Apps and Tutorials

## GETTING STARTED

### Install the SDK

Releases page:

`https://github.com/getchui/trueface-sdk/releases`

Install:

`pip install wheel_url`

ex:

`sudo pip install https://github.com/getchui/offline_sdk/releases/download/0.61/trueface-0.0.6.1-cp27-cp27mu-linux_x86_64.whl`

Download models:

Face Detect

`wget https://github.com/getchui/offline_sdk/releases/download/models-latest/fd_model.zip`

TF-Lite

`wget https://github.com/getchui/offline_sdk/releases/download/models-latest/model-lite.zip`

TFV2 (recommended)

`wget https://github.com/getchui/offline_sdk/releases/download/models-latest/model-tfv2.zip`

TFV3

`wget https://github.com/getchui/offline_sdk/releases/download/models-latest/model-tfv3.zip`


### FR With Blur Demo

```
Step 1. Set your Token
export TF_TOKEN=your_token


Step 2. Create Collection
python create_collection.py

Step 3. Run Demo
python fr_with_blur.py

```

### Working with Different Capture Devices
USB Cameras

`vcap = VideoStream(src=0).start()`

IP Cameras, RTSP, HLS, MJPEG
Use the stream url as the source:

`vcap = VideoStream(src="rtsp://192.168.1.11:554/11").start()`

