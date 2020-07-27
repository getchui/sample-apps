import os
import flask
from sqlalchemy import Column, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.types import BLOB, Integer, Interval, String, DateTime, BIGINT, Text, Boolean, Float
from sqlalchemy_utils import ChoiceType, URLType, database_exists, create_database
from functools import wraps
from flask import Flask, render_template, request, session, request, redirect
from flask import url_for
import uuid
import json
import jwt
from datetime import datetime
import time
import base64
import uuid
import cv2
import numpy as np


app = Flask(
    __name__,
    static_folder='./frontend/build/static',
    template_folder='./frontend/build')

@app.route('/manifest.json')
def manifest():
    """handles home"""
    return render_template('manifest.json')

@app.route('/')
def home():
    """handles home"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=False)