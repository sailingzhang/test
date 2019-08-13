#!/bin/bash
# exec &>>/var/log/xxxxx.log
# set -x 
echo "hello face"
cd /opt/tvu/R/iMatrix/facerecognition
/opt/tvu/aienv/bin/python  /opt/tvu/R/iMatrix/facerecognition/local_face_server.py 40051 /usr/share/nginx/localFaceData
