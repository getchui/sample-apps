FROM ubuntu:18.04

COPY . /

RUN sh install.sh

RUN pip3 --no-cache-dir install -r reqs.txt

EXPOSE 5000

CMD ["pm2-runtime", "start", "process.yml" ]
