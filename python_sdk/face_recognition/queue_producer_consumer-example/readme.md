
# Overview
Microservices allow you to optimize resources, easily scale services, and run distributed applications.

This example shows you can run face recognition as two services, a producer running face detect, and a consumer consuming the faces and running identify.

The example utilizes batch_identify, batching predictions through our engine can achieve significantly higher processing speeding compared with single image inference. The example shows processing 100 faces at time in every batch with a 100+ layer model!



## Install required pip packages
`pip install -r requirements.txt`



## Install Trueface sdk
`sudo pip install https://github.com/getchui/offline_sdk/releases/download/0.61/trueface-0.0.6.1-cp27-cp27mu-linux_x86_64.whl`


## If using a GPU, install mxnet gpu for your CUDA version

`sudo pip install mxnet-cu101`

## Start a redis instance with docker
`docker run -di --name redis --restart always -p 6379:6379 redis`

## Download Models

`wget https://github.com/getchui/offline_sdk/releases/download/models-latest/fd_model.zip`


## run producer
`python producer.py`

this allows run face detect in parallel way extracting faces and pushing them to the queue for processing.


## run consumer
`python consumer.py`

this runs the consumer which retrives the faces from the in memory queue and runs identify.

## how to create a collection
Use the python create collection in the SDK.
https://docs.trueface.ai/docs/the-sdk

