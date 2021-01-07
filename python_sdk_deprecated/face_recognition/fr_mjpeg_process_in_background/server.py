'''Server Code'''
import GPUtil
from functools import wraps
from flask import Flask, render_template, request, session, request, redirect, Response
from flask import url_for
import uuid
import json
import requests
import psutil
import time
from utils import get_ip_address
import os
from jetson_stats import jetson_gpu_usage

app = Flask(__name__)

@app.route('/usage')
def get_usage():
	gpu_info = []
	#this block is used on desktop computers with nvidia-smi avaliable
	GPUs = GPUtil.getGPUs()
	for gpu in GPUs:
		print gpu.temperature
		print(' {0:2d} {1:3.0f}% {2:3.0f}%'.format(gpu.id, gpu.load*100, gpu.memoryUtil*100))
		gpu_info.append({"id":gpu.id, "load":gpu.load*100, "memory":gpu.memoryUtil*100})

	#this is used on jetsons when there's no nvidia-smi and CPU/GPU share RAM memory
	#only one GPU on jetson, no need for a loop
	# gpu_usage = jetson_gpu_usage()
	# gpu_info.append({"id":0, "load":gpu_usage, "memory": psutil.virtual_memory().percent})

	cpu_info = {"cpu":psutil.cpu_percent(), "ram": psutil.virtual_memory().percent}
	return json.dumps({
		"timestamp":int(time.time()), 
		"gpu_info":gpu_info, 
		"cpu_info":cpu_info}), 200, {'Content-Type': 'application/json'}



@app.route('/test_webhook', methods=['POST'])
def test_webhook():
	return json.dumps({"success":True}), 200, {'Content-Type': 'application/json'}


@app.route('/highspeed', methods=['POST'])
def highspeed():
	os.system("cd ~/ && sudo ./jetson_clocks.sh")
	return json.dumps({"success":True}), 200, {'Content-Type': 'application/json'}


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def home(path):
	if path == '':
		page = 'index.html'
	else:
		page = path
	return render_template(page)

if __name__ == '__main__':
    app.run(debug=False, host= '0.0.0.0')