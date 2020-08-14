FROM node:14.7.0-stretch

COPY ./frontend /frontend
COPY ./main.py /

RUN cd /frontend && npm install && npm audit fix && npm run-script build && cd ..

RUN apt-get update && apt-get -y install python3-pip && pip3 install flask SQLAlchemy SQLAlchemy-Utils jwt numpy

CMD ["/main.py"]
ENTRYPOINT ["python3"]