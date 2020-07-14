#!/bin/bash

echo Trueface.ai AgeBox Sample App

cd server
npm install
sudo docker-compose build
sudo docker-compose up
