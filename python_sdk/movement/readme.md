## Get Started
To Download tf-yolov3 or mobilenet model:

```
wget https://github.com/getchui/offline_sdk/releases/download/models-latest/tf-object_detection-mobilenet.zip && unzip tf-object_detection-mobilenet.zip
```
or 

```
wget https://github.com/getchui/offline_sdk/releases/download/models-latest/tf-object_detection-yolov3.zip && unzip tf-object_detection-yolov3.zip
```

## install Trueface sdk
`sudo pip install https://github.com/getchui/offline_sdk/releases/download/0.61/trueface-0.0.6.1-cp27-cp27mu-linux_x86_64.whl`

Get your platform wheel url from the following page:<br/>


### Run Demo
```
export TF_TOKEN={your_token}
python demo.py
```


### How do I use this?

1. You can divide your frame into regions, and you can use the points stored in the tracked object to check the regions visited by the object with a function like cv2.pointPolygonTest
2. You can draw the movement live or in retrospect to visualize movement through a location.