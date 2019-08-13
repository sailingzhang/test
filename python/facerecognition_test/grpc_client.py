# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The Python implementation of the GRPC helloworld.Greeter client."""

from __future__ import print_function


import sys
import os
# work_dir ="/home/sailingzhang/winshare/develop/source/mygit/facerecognition"
work_dir =os.getcwd()
sys.path.append(work_dir)
sys.path.append(work_dir+"/faceproto/")
sys.path.append(work_dir+"/facenet/src/")

import grpc

from google.protobuf.json_format import MessageToJson
from faceproto import faceproto_pb2,faceproto_pb2_grpc
from log  import log_init
import logging
import uuid
import time


def run():
    logging.debug("enter")
    addr = sys.argv[1]
    channel = grpc.insecure_channel(addr)
    stub = faceproto_pb2_grpc.FaceIdentyStub(channel)

    # response = stub.HelloFace(faceproto_pb2.HelloFaceReq(ask="this is python"))
    # print("Greeter client received: " + response.response)

    img_paths=["pic/to_picface_0.png","pic/to_picface_1.png","pic/to_picface_2.png","pic/cropped.jpg"]
    # # img_paths=["to_picface_0.png","to_picface_1.png"]
    # # img_paths=["to_picface_2.png"]
    # img_paths=["to_picface_3.png"]
    # initReq = faceproto_pb2.InitFaceReq(peerId="111111")
    # response =stub.InitFace(initReq)
    # logging.debug("init response={}".format(response))


    # compareReq = faceproto_pb2.CompareReq()
    # for i in range(len(img_paths)):
    #     path = img_paths[i]
    #     face = compareReq.faces.add()
    #     f = open(path,"rb+")
    #     data = f.read()
    #     f.close()
    #     face.Id = str(i)
    #     face.facePic=data
    # rsp = stub.Compare(compareReq)
    # logging.debug("compare rsp={}".format(rsp))
    # # rspjson = MessageToJson(rsp,True)
    # # logging.debug("compare rspjson={}".format(rspjson))
    # # return

    # findReq = faceproto_pb2.FindSimilarFaceReq()
    # findReq.peerId="1111"
    # findReq.threshold  =0.98
    # for i in range(len(img_paths)):
    #     path = img_paths[i]
    #     face = findReq.faces.add()
    #     f = open(path,"rb+")
    #     data = f.read()
    #     f.close()
    #     face.Id = str(uuid.uuid1())
    #     face.facePic=data
    # # rsp = stub.FindSimilarFace(findReq)
    # rsp = stub.FindSimilarHistoryFace(findReq)
    # rspjson = MessageToJson(rsp,True)
    # logging.debug("findsimilar rspjson={}".format(rspjson))


    initReq = faceproto_pb2.InitFaceReq()
    initReq.peerId ="testpeerid"
    rsp = stub.InitFace(initReq)
    rspjson = MessageToJson(rsp,True)
    logging.debug("findsimilar rspjson={}".format(rspjson))


    
    # findReq = faceproto_pb2.FindSimilarFaceReq()
    # findReq.peerId="1111"
    # findReq.threshold  =0.1
    # findReq.timestamp = 123456
    # for i in range(len(img_paths)):
    #     path = img_paths[i]
    #     face = findReq.faces.add()
    #     f = open(path,"rb+")
    #     data = f.read()
    #     f.close()
    #     face.Id = str(uuid.uuid1())
    #     face.facePic=data
    # # rsp = stub.FindSimilarFace(findReq)
    # rsp = stub.FindSimilarHistoryFace(findReq)
    # rspjson = MessageToJson(rsp,True)
    # logging.debug("findsimilar rspjson={}".format(rspjson))




    detectimg ="pic/test.jpg"
    detectReq = faceproto_pb2.DetectReq()
    detectReq.id=str(int(round(time.time() * 1000)))
    f = open(detectimg,"rb+")
    data = f.read()
    f.close()
    detectReq.pic = data
    rsp = stub.Detect(detectReq)
    logging.debug("detect rsp={}".format(rsp))



    # predictpic ="5e2c76669f6993c67db06ddcf5b1c9c6.jpg"
    predictpic ="pic/timg.jpg"
    # predictpic ="heying.jpg"
    predictReq = faceproto_pb2.PredictReq()
    predictReq.id=str(int(round(time.time() * 1000)))
    f = open(predictpic,"rb+")
    data = f.read()
    f.close()
    predictReq.pic = data
    rsp = stub.Predict(predictReq)
    logging.debug("predict rsp={}".format(rsp))
    



if __name__ == '__main__':
    log_init.log_init("/tmp/grpc_client.log")
    run()
