FROM node:10-alpine
WORKDIR /usr/src/app
COPY ./server/package*.json ./
RUN npm install
RUN npm install
COPY . .
COPY ./server/. .
WORKDIR /usr/src/app/frontend
COPY ./frontend/package*.json ./
RUN npm install
COPY ./frontend/. .
RUN npm run build
RUN mv /usr/src/app/frontend/build /usr/src/app/build
WORKDIR /usr/src/app
EXPOSE 4000
CMD ["node", "server.js"]