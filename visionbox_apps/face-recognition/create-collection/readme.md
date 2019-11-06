## Visionbox Start Command
sudo docker run -p 8085:8080 -e "TOKEN=your_token"  -v "$(pwd)"/collections:/collections trueface/visionbox:latest

## install dependencies for demo.py
`pip install -r requirements.txt`

## RUN
`python3 demo.py`

to use form data instead of json change

`r = requests.post(base_url+'/enroll', json=data)`

to 

`r = requests.post(base_url+'/enroll', data=data)`

