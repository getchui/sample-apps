#!/bin/bash

echo Trueface.ai AgeBox Sample App

cd frontend
npm install
npm run build
cp -r ./build/ ../server/build
cd ../server
docker-compose up