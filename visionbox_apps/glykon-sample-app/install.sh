#!/bin/sh
export DEBIAN_FRONTEND="noninteractive"

apt-get update
apt-get -yq --no-install-recommends install build-essential
apt-get -yq --no-install-recommends install apt-utils
apt-get -yq --no-install-recommends install curl
apt-get -yq --no-install-recommends install python-dev
apt-get -yq --no-install-recommends install python3-dev
apt-get -yq install cmake
apt-get -yq install --no-install-recommends \
      rsync \
      software-properties-common \
      unzip \
      clang \
      pkg-config \
      python-pip \
      python3-dev \
      python3-setuptools \
      python3-sklearn  \
      python3-scipy \
      node-gyp \
      rsync \
      npm \
      unzip \
      pkg-config \
      libjpeg-dev \
      libpng-dev \
      libtiff-dev \
      libavformat-dev \
      libhdf5-dev \
      libpq-dev \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libzmq3-dev \
      pkg-config \
      libblas-dev \
      liblapack-dev \
      rsync \
      libssl-dev \
      software-properties-common \
      zip \
      libgtk2.0-dev \
      zlib1g-dev \
      libmysqlclient-dev \
      cmake \
      wget curl vim dialog net-tools nmap netcat procps coreutils grep iproute2 python-pudb python-pytest

apt-get clean

curl -sL https://deb.nodesource.com/setup_13.x | bash -
apt-get install -yq --no-install-recommends nodejs
apt-get install -yq --no-install-recommends npm
apt-get install -yq  --no-install-recommends python3-pip
apt-get install -yq  --no-install-recommends libgtk2.0-dev


npm i -g pm2
python3 --version

pip3 install setuptools
