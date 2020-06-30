import os
import flask
from functools import wraps
from flask import Flask, render_template, request, session, request, redirect
from flask import url_for
import json
import base64
import requests

app = Flask(__name__)

def parse_payload(request):
    try:
        if request.method == 'POST' or request.method == 'PUT':
                if request.data:
                    return  request.get_json(force=True)
                elif request.files:
                    return request.files
                else:
                    return request.form.to_dict()
        if request.method == 'GET':
                return request.args

    except Exception as e:
        print(e)

@app.route('/manifest.json')
def manifest():
    """handles home"""
    return render_template('manifest.json')

@app.route('/post-video', methods=["POST"])
def post_video():
    """post video"""
    video = parse_payload(request)['video']
    dvideo = base64.b64decode(video.split(",")[1])
    with open("temp.webm", "wb") as f:
        f.write(dvideo)

    base = os.environ["API_PATH"]
    url = "%s/run" % base
    data = {
            "video":open("./temp.webm", "rb")
    }
    r = requests.post(url, files=data)
    print(r.json())

    return json.dumps(r.json()), 200, {'Content-Type': 'application/json'}

@app.route('/')
def home():
    """handles home"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=False)