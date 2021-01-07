#!/usr/bin/env bash
test -e spoofv4 || (wget https://github.com/getchui/offline_sdk/releases/download/models-latest/spoofv4.zip && unzip spoofv4.zip)
