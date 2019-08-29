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

"""The Python implementation of the GRPC helloworld.Greeter server."""

import sys
import os

work_dir =os.getcwd()
sys.path.append(work_dir)
sys.path.append(work_dir+"/faceproto/")
# sys.path.append(work_dir+"/facenet/src/")
sys.path.append(work_dir+"/facenet/")
sys.path.append(work_dir+"/facenet/align")
# sys.path.append(work_dir+"/mtcnnpytorch")

from concurrent import futures
import time

import grpc

from faceproto import faceproto_pb2,faceproto_pb2_grpc
from log  import log_init
import logging



import argparse
import tensorflow as tf
import numpy as np
import facenet
from log import log_init
import logging
# import align.detect_face
import random
from time import sleep
from io import BytesIO 
from PIL import Image, ImageFilter
import imageio
from scipy import misc
# from myface.mtcnn_facenet import detectface,facenet_ebeding,myclassify,myclassify2,TvuFace
from mtcnn_facenet import detectface,facenet_ebeding,myclassify,myclassify2,TvuFace

from classifier import NeurosFaceClassifier,NeurosFaceClassifierStd
from sklearn.neighbors import KNeighborsClassifier 

import threading
import gc
# import string
import psutil

from mtcnnpytorch.src.detector import detect_faces,pytorchDetect
_ONE_DAY_IN_SECONDS = 60 * 60 * 24




def pretectFunBak():
    logging.debug("enter")
    pid = os.getpid()
    command =r"ps aux | grep -v grep | awk -F ' ' '{print $2,$4}'|grep "+str(pid)+r"| awk -F ' ' '{print $2}'"
    with os.popen(command) as f:
        res = f.read()
        resStr=""
        for c in res:
            if (c>'0' and c <'9') or c=='.':
                resStr += c
        logging.info("resStr={},len(resStr)".format(resStr,len(resStr)))
        m= eval(res)#float(res)
        # logging.info("m={}".format(m))
        logging.info("type(res)={},len(res)={},command={},ress={},m={}".format(type(res),len(res),command,res,m))
    

def protectFun():
    p1=psutil.Process(os.getpid())
    m = p1.memory_percent()
    allmem = psutil.virtual_memory()
    allUsePercent = allmem.percent
    allAvailable = allmem.available/1024/1024
    logging.info("m={},allUsePercent={},allAvailable={}".format(m,allUsePercent,allAvailable))
    if m > 40 or allAvailable < 500:
        logging.error("stop gServer")
        gServer.stop(0)
        os._exit(1)

def timer():
    while True:
        logging.info("enter")
        time.sleep(60)
        protectFun()
def embed_performance_test():
    logging.debug("enter")
    faceEmd = facenet_ebeding(work_dir+"/../sailingzhang/embed_model")
    img_list = []
    for dirpath, dirnames, filenames in os.walk(work_dir+"/../tmp/ml_pic/to_compare"):
        for filename in filenames:
            fullname = os.path.join(dirpath, filename)
            img_list.append(fullname)

    for it in range(5):
        logging.debug("test begin")    
        embs=faceEmd.embed_paths(img_list[it:it+1])
        # logging.debug("len(embs)={}".format(len(embs)))
        logging.debug("begin compare")
        for i in range(10):
            print("\n{} ".format(i),end='')
            for j in range(10):
                dist = np.sqrt(np.sum(np.square(np.subtract(embs[0,:], embs[0,:]))))
                print('  %1.4f  ' % dist, end='')
        logging.debug("test end")  

# model = NeurosFaceClassifierStd(6000,1)
class FaceGreeter(faceproto_pb2_grpc.FaceIdentyServicer):
    def __init__(self):
        logging.debug("init begin")
        self.lock = threading.Lock()
        ##############################################
        model = KNeighborsClassifier(n_neighbors=1)
        datadir = sys.argv[2]
        logging.debug("datadir={},type(datadir)={},len(datadir)={}".format(datadir,type(datadir),len(datadir)))
        self.tvuface = TvuFace(work_dir,datadir,model)
        # self.tvuface.fit()
        ##############################################
        self.embedFaceNum = 0
        self.embedFaceMaxLimitNum = 2000
        logging.debug("init end")

    def _predect_map(self,string_id,detMap):
        rsp = faceproto_pb2.PredictRsp()
        rsp.id = string_id
        faceindex = 0
        with self.lock:
            for obj in detMap.values():
                faceindex+=1
                faceojb= obj[1]
                facebox =obj[0]
                predictList = self.tvuface.predict([faceojb.emb])
                faceid = faceojb.faceid
                logging.debug("predictList={},faceid={},obj={}".format(predictList,faceid,obj))
                addface = rsp.predictFaces.add()
                addface.faceid = faceid
                addface.left = facebox[0]
                addface.top = facebox[1]
                addface.width = facebox[2]
                addface.height = facebox[3]
                predict = predictList[0]            
                if predict is None:
                    logging.debug("faceid={} no predict,begin addperson".format(faceid))
                    addpersonid = "P-"+string_id+"-"+str(faceindex)
                    logging.debug("no predict ,addpersonid={}".format(addpersonid))
                    self.tvuface.addFaceToPerson(addpersonid,faceojb.faceid,faceojb.alingned_pic,faceojb.emb)
                    self.tvuface.fit()
                    addface.personid = addpersonid
                    addface.newperson = True
                    addface.confidence = 0
                    logging.info("new_personid={},addfaceid={},width={},height={}".format(addpersonid,obj[1].faceid,addface.width,addface.height))
                else:
                    personid = predict[0]
                    dist = predict[2]
                    addface.personid = personid
                    addface.newperson = False
                    addface.confidence = dist
                    logging.info("old_personid={},faceid={},width={},height={}".format(personid,faceid,addface.width,addface.height))
                    self.tvuface.addFaceToPerson(personid,faceojb.faceid,faceojb.alingned_pic,faceojb.emb)
                    self.tvuface.fit()
            logging.debug("normal,exit")
            return rsp
    def _initFace(self):
        logging.info("enter")
        with self.lock:
            self.tvuface.clear(KNeighborsClassifier(n_neighbors=1))
            logging.debug("exit")
        gc.collect()

    def InitFace(self,request,context):
        logging.info("enter")
        # with self.lock:
        #     self.tvuface.clear(KNeighborsClassifier(n_neighbors=1))
        #     logging.debug("exit")
        self._initFace()
        return faceproto_pb2.InitFaceRsp(error=0,error_message="no error")
    def FindSimilarHistoryFace(self,request,context):
        logging.debug("enter,facenum={}".format(len(request.faces)))
        imgList=[]
        idList =[]
        rsp = faceproto_pb2.FindSimilarFaceRsp()
        threshold = request.threshold
        peerid = request.peerId
        faces  = request.faces
        if 0 == len(faces):
            logging.info("timestamp={},face number is 0".format(request.timestamp))
            return rsp
        for face in faces:
            logging.debug("peerid={},threshold={},face.Id={}".format(peerid,threshold,face.Id))
            bi = BytesIO(face.facePic)
            image = Image.open(bi)
            # image =misc.fromimage(image)
            image =imageio.imread(face.facePic)
            imgList.append(image)
            idList.append(face.Id)
        dectMap = self.tvuface.detectAlinedImgs(idList,imgList)

        rsp = self._predect_map(str(request.timestamp),dectMap)

        self.embedFaceNum += len(dectMap)
        if self.embedFaceNum > self.embedFaceMaxLimitNum:
            logging.info("facenumber limit,maxlimitnum={}".format(self.embedFaceMaxLimitNum))
            self.embedFaceNum =0
            self._initFace()
        return rsp
        

    def Detect(self,request,context):
        logging.debug("enter")
        rsp = faceproto_pb2.PredictRsp()
        rsp.id = request.id
        if 0 == len(request.pic):
            logging.debug("pic is 0,exit")
            return rsp
        bi = BytesIO(request.pic)
        image = Image.open(bi)
        # image.save(rsp.id+".jpg")
        # image =misc.fromimage(image)
        image =imageio.imread(request.pic)
        faceMinSize =min(image.shape[1],image.shape[0])//20
        logging.debug("begin detectImg,faceMinSize={},misc'shape={}".format(faceMinSize,image.shape))
        dectMap = self.tvuface.detectImg(request.id,image,max(faceMinSize,20))
        logging.debug("detMap={}".format(dectMap))
        if not dectMap:
            logging.debug("empty detmap,exit")
            return rsp
        
        faceindex =0
        for obj in dectMap.values():
            faceindex+=1
            faceojb= obj[1]
            facebox =obj[0]
            addface = rsp.predictFaces.add()
            addface.faceid = faceojb.faceid
            addface.left = facebox[0]
            addface.top = facebox[1]
            addface.width = facebox[2]
            addface.height = facebox[3]
        return rsp

        

    def Predict(self,request,context):
        logging.debug("enter")
        rsp = faceproto_pb2.PredictRsp()
        rsp.id = request.id
        if 0 == len(request.pic):
            logging.debug("pic is 0,exit")
            return rsp
        bi = BytesIO(request.pic)
        image = Image.open(bi)
        # image.save(rsp.id+".jpg")
        image =misc.fromimage(image)
        faceMinSize =min(image.shape[1],image.shape[0])//20
        logging.debug("begin detectImg,faceMinSize={},image'shape={}".format(faceMinSize,image.shape))
        dectMap = self.tvuface.detectImgAndEmbed(request.id,image,max(faceMinSize,20))
        logging.debug("detMap={}".format(dectMap))
        if not dectMap:
            logging.debug("empty detmap,exit")
            return rsp

        rsp = self._predect_map(request.id,dectMap)

        self.embedFaceNum += len(dectMap)
        if self.embedFaceNum > self.embedFaceMaxLimitNum:
            logging.info("facenumber limit,maxlimitnum={}".format(self.embedFaceMaxLimitNum))
            self.embedFaceNum =0
            self._initFace()
        return rsp


def faceServe(port):
    logging.debug("enter")
    global gServer
    gServer = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    faceproto_pb2_grpc.add_FaceIdentyServicer_to_server(FaceGreeter(), gServer)
    gServer.add_insecure_port('[::]:'+port)
    # gServer.add_insecure_port('10.12.23.213:50051')
    gServer.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        gServer.stop(0)



def test():
    detect = detectface()
    return
    while True:
        bTime = time.time()
        logging.debug("tensorflow begin")
        ret = detect.defect_path("pic/timg.jpg")
        eTime = time.time()
        logging.debug("tensorflow end,costTime={}".format(eTime-bTime))
        logging.debug("ret={}".format(ret))
    
def test_remoteserver():
    logging.info("enter")
    a = tf.constant(1.0)
    b = a+2
    c = a*3
    with tf.Session("grpc://127.0.0.1:2222") as sess:
        logging.info("c={}".format(sess.run(c)))

    

def pytorch_test():
    image = Image.open("pic/timg.jpg")
    # image = Image.open("pic/test.jpg")
    pytorchD = pytorchDetect()
    while True:
        bTime = time.time()
        logging.info("pytorch begin")
        # bounding_boxes, landmarks = detect_faces(image)
        bounding_boxes, landmarks = pytorchD.detect_faces(image)
        eTime = time.time()
        logging.info("pytorch end,costTime={}".format(eTime-bTime))
        logging.info("bounding_boxes={}".format(bounding_boxes))

if __name__ == '__main__':
    port = sys.argv[1]
    log_init.log_init("p_local_face_server_"+port+".log")
    logging.info("start gServer")
    # pytorch_test()
    test()
    # threading.Thread(target=timer).start()
    # faceServe(port)
