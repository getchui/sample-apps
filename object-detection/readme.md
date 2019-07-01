
# Object Detection Tutorials

`COCO mAP TFYolov3: 36.0`
`COCO mAP TFMobileNet: 28.6`


## Get Started
To Download yolov3 or mobilenet model

`sh download_mobilenet.sh`
`sh download_tfyolov3.sh`

### install Trueface sdk
`sudo pip install https://github.com/getchui/offline_sdk/releases/download/0.61/trueface-0.0.6.1-cp27-cp27mu-linux_x86_64.whl`

Get your platform wheel url from the following page:\s
https://github.com/getchui/offline_sdk/releases/download/0.61/trueface-0.0.6.1-cp27-cp27mu-linux_x86_64.whl

### Run Demo
run `export TF_TOKEN={your_token}`  to define your token then run `python demo.py`.