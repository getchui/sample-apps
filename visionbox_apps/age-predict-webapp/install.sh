#!/bin/bash

echo Trueface.ai AgeBox Sample App

cd frontend
npm install
npm run build
cp --parents ./build/ ../server -r
cd ../server
docker-compose up