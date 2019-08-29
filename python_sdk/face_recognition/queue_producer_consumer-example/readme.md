
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