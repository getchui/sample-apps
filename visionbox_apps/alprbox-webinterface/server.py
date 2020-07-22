'''Server Code'''
from functools import wraps
from flask import Flask, render_template, request, session, request, redirect, Response
from flask import url_for
import cv2
import numpy as np
import flask
import uuid
import json
import requests
import base64
import time
import os

app = Flask(__name__)

def download_image(url):
        r = requests.get(url, stream=True)
        return r.content

def parse_payload(request):
    try:
        if request.data:
            return  request.get_json(force=True)
        elif request.files:
            return request.files
        else:
            return request.form.to_dict()
    except Exception as e:
        print(e)

def get_image(data):
    if 'url' in data:
            return cv2.imdecode(np.fromstring(download_image(data['url']), np.uint8), 1)
    elif 'image' in data:
            print(type(data['image']))
            if type(data['image']) == unicode:
                img = base64.b64decode(data['image'].split(",", 2)[1])
            elif  type(data['image']) == werkzeug.datastructures.FileStorage:
                img = data['image'].read()
            img = cv2.imdecode(np.fromstring(img, np.uint8), 1)
            return img

@app.route('/process-image', methods=["POST"])
def process_image():
	img = request.json['image'].split(',', 2)[1]
	data = {
	  "image":img,
	}
	r = requests.get(os.environ["API_URL"]+'/predict', json=data)
	print(r.json())
	
	return flask.jsonify(r.json()), 200, {"Content-Type":"application/json"}  
     
@app.route('/process-url', methods=["POST"])
def process_url():
	url = request.json['url']
	data = {
	    "url":url,
	    "make_model":False,   # detect the make and model of the car
	    "car_detect":True,   # return the location of the car in the image
	    "color_detect":False, # return the colors of the car
		}
	r = requests.get(os.environ["API_URL"]+'/predict', params=data)
	print(r.content)
	return flask.jsonify(r.json()), 200, {"Content-Type":"application/json"}

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def home(path):
	if path == '':
		page = 'index.html'
	else:
		page = path
	return render_template(page)


if __name__ == '__main__':
    app.run(debug=True, host= '0.0.0.0')
