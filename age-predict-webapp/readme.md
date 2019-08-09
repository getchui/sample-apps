# Age Prediction Webapp

## Step 1

Launch docker image for agebox 
```
docker run -p 9000:8080 -e "TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbW90aW9uIjp0cnVlLCJmciI6dHJ1ZSwicGFja2FnZV9pZCI6bnVsbCwiZXhwaXJ5X2RhdGUiOiIyMDE5LTEwLTE2IiwidGhyZWF0X2RldGVjdGlvbiI6dHJ1ZSwibWFjaGluZXMiOiI1IiwiYWxwciI6dHJ1ZSwibmFtZSI6Ik5lemFyZSBDaGFmbmkiLCJ0a2V5IjoibmV3IiwiZXhwaXJ5X3RpbWVfc3RhbXAiOjE1NzExODQwMDAuMCwiYXR0cmlidXRlcyI6dHJ1ZSwidHlwZSI6Im9ubGluZSIsImVtYWlsIjoibmNoYWZuaUBnbWFpbC5jb20ifQ.UQIpPpxeCdACm12hagzxI2yTcMEJbdfKn3moZOEpCl8" trueface/agebox:latest
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