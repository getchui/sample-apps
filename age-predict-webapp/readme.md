# Age Prediction Webapp

## Step 1

Launch docker image for agebox 
```
docker run -p 9000:8080 -e "TOKEN=yourtoken" trueface/agebox:latest
```

Make sure docker opens port 9000.

## Step 2

Install all the dependencies

```
cd age-predict-webapp
npm install
cd server
npm install 
cd ../webapp
npm install
```

## Step 3

Launch the app, make sure you are at the **age-predict-webapp** folder. 
```
npm start
```

Open browser and goto ```http://localhost:3000``` to access the app.