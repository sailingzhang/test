"""Performs face alignment and stores face thumbnails in the output directory."""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from scipy import misc
import imageio
import sys


print("sys.path=",sys.path)
import os
import argparse
import tensorflow as tf
import numpy as np
# import facenet
from facenet import facenet,align
from log import log_init
import logging

from align  import detect_face
# from facenet import align


import random
from time import sleep,time

import math
import pickle
from sklearn.svm import SVC,LinearSVC
from sklearn.preprocessing import StandardScaler

from  classifier import  NeurosFaceClassifier
from dataset import trainDataBase,LocalDataManger
import uuid
import logging
# import time
import threading
# from mtcnnpytorch.src.get_nets import create_pytorch_mtcnn

# log_init.log_init("/tmp/mtcnn_facenet.log")
# logging.debug("mtcnn module ok")


class detectface:
    def __init__(self):
        logging.info("detectface init begin")
        self.minsize = 20 # minimum size of face
        self.threshold = [ 0.7, 0.8, 0.9 ]  # three steps's threshold
        self.factor = 0.705 # scale factor
        self.gpu_memory_fraction = 0.4
        self.cropped_image_size = 80
        self.count =0


        

        # with tf.Graph().as_default():
        #     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        #     with sess.as_default():
        #         self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(sess, None)


        self.grpah = tf.Graph().as_default()
        # with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction,allow_growth = True)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False,device_count={'GPU':0, 'CPU':2}))
        # self.sess = tf.Session("grpc://127.0.0.1:2222")
        # self.sess = tf.Session()

        cpu_num = 4
        tvuconfig = tf.ConfigProto(device_count={"CPU": cpu_num},inter_op_parallelism_threads = cpu_num,intra_op_parallelism_threads = cpu_num,log_device_placement=True)
        self.sess = tf.Session(config = tvuconfig)
        self.sess.as_default()
        # self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, None)
        self.pnet, self.rnet, self.onet = align.detect_face.create_tvu_mtcnn(self.sess, None)
        # self.pnet, self.rnet, self.onet = align.detect_face.create_tvu_mtcnn_pb(self.sess,"/tmp/myfirst.pb")
        
        tf.get_default_graph().finalize()

        # self.pnet, self.rnet, self.onet =create_pytorch_mtcnn()
        
        logging.info("detectface init end")
    def reload(self):
        logging.debug("enter")
        if self.sess:
            self.sess.close()
            self.sess =None
        # self.grpah = tf.Graph().as_default()
        # with tf.Graph().as_default():
        # tf.reset_default_graph()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # self.sess.as_default()
        # self.pnet, self.rnet, self.onet = align.detect_face.create_mtcnn(self.sess, None)
        # tf.get_default_graph().finalize()

    def defect_path(self,image_path):
        try:
            # img = misc.imread(image_path)
            img = imageio.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            logging.error(errorMessage)
            return
        return self.detect_img(img)

    def detect_img(self,img,faceMinsize=0):
        picwidth = img.shape[1]
        picheight = img.shape[0]
        logging.debug("img.ndim ={},picwidth={},picheight={}".format(img.ndim,picwidth,picheight))
        if img.ndim<2:
            logging.error('Unable to align "%s"' % image_path)
            return
        if img.ndim == 2:
            img = facenet.to_rgb(img)
            logging.debug("begin rgb")    
            img = facenet.to_rgb(img)  
            logging.debug("end rgb")
        img = img[:,:,0:3]
        if 0 == faceMinsize:
            faceMinsize = self.minsize
        bounding_boxes, _ = align.detect_face.detect_face(img, faceMinsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        logging.debug("faceminisize={},bouding's shape={},bounding={}".format(faceMinsize,bounding_boxes.shape,bounding_boxes))
        for i in range(bounding_boxes.shape[0]):
            for j in range(bounding_boxes.shape[1]):
                if bounding_boxes[i][j] < 0:
                    logging.debug("bounding_boxes[{}][{}] < 0,value={}".format(i,j,bounding_boxes[i][j]))
                    bounding_boxes[i][j] = 0
            if bounding_boxes[i][2] > picwidth:
                logging.info("adjust width")
                bounding_boxes[i][2] = picwidth
            if bounding_boxes[i][3] > picheight:
                logging.info("adjust height")
                bounding_boxes[i][3] = picheight
        return bounding_boxes

    def truncate(self,img,bounding_box):
        logging.debug("bounding_box={}".format(bounding_box))
        det = np.squeeze(bounding_box)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = det[0]
        bb[1] = det[1]
        bb[2] = det[2]
        bb[3] = det[3]
        center = 30
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        # cropped = img[bb[1]+center:bb[3]-center,bb[0]+center:bb[2]-center,:]
        # scaled = misc.imresize(cropped, (self.cropped_image_size, self.cropped_image_size), interp='bilinear')
        scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
        # scaled = misc.imresize(cropped, (bb[3]-bb[1], bb[2]-bb[0]), interp='bilinear')
        logging.debug("crop'shape={},scaled'shape={}".format(cropped.shape,scaled.shape))
        # misc.imsave("cropped.jpg", scaled)
        return scaled

        
class facenet_ebeding:
    def __init__(self,modelpath):
        self.modelpath=modelpath
        self.emb =self.get_emb_img()
        # threading.Thread(target=facenet_ebeding.get_emb_img,args=(self,)).start()

    # def get_emb_img(self):
    #     with tf.Graph().as_default():
    #         sess = tf.Session()
    #         with sess.as_default():
    #             logging.info("really loading emb")
    #             facenet.load_model(self.modelpath)
    #             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    #             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    #             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    #             def embed_base(aligned_img_list):
    #                 logging.debug("begin embed,aligned_emd_list'len={}".format(len(aligned_img_list)))
    #                 prewhitened_list =[]
    #                 for i in range(len(aligned_img_list)):
    #                     logging.debug("type(aligned_img[{}])={}".format(i,type(aligned_img_list[i])))
    #                     prewhitened = facenet.prewhiten(aligned_img_list[i])
    #                     prewhitened_list.append(prewhitened)
    #                 prewhitened_stack = np.stack(prewhitened_list)

    #                 feed_dict = { images_placeholder: prewhitened_stack, phase_train_placeholder:False }
    #                 logging.debug("begin run embeddings")
    #                 emb = sess.run(embeddings, feed_dict=feed_dict)
    #                 logging.debug("end run embedding, type(emb)={},emb.shape={}".format(type(emb),emb.shape))
    #                 return emb
    #             self.emb = embed_base
    #             tf.get_default_graph().finalize()
    #             return embed_base


    def get_emb_img(self):
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                logging.info("really loading emb")
                facenet.load_model(self.modelpath)
                self.sess = sess
                # images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                # embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                # phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                def embed_base(aligned_img_list):
                    logging.debug("begin embed,aligned_emd_list'len={}".format(len(aligned_img_list)))
                    prewhitened_list =[]
                    for i in range(len(aligned_img_list)):
                        logging.debug("type(aligned_img[{}])={}".format(i,type(aligned_img_list[i])))
                        prewhitened = facenet.prewhiten(aligned_img_list[i])
                        prewhitened_list.append(prewhitened)
                    prewhitened_stack = np.stack(prewhitened_list)

                    feed_dict = { "input:0": prewhitened_stack, "phase_train:0":True }
                    logging.debug("begin run embeddings")
                    emb = sess.run("embeddings:0", feed_dict=feed_dict)
                    logging.debug("end run embedding, type(emb)={},emb.shape={}".format(type(emb),emb.shape))
                    return emb
                self.emb = embed_base
                # tf.get_default_graph().finalize()
                return embed_base


    def embed(self,aligned_img_list):
        # while self.emb is None:
        #     logging.info("loading emb,wait")
        #     sleep(5)
        return self.emb(aligned_img_list)
    def embed_paths(self,alined_img_paths):
        # while self.emb is None:
        #     logging.info("loading emb,wait")
        #     sleep(5)
        img_list=[]
        for i in range(len(alined_img_paths)):
            logging.debug("begin read img={}".format(alined_img_paths[i]))
            img = imageio.imread(alined_img_paths[i])
            img_list.append(img)
        return self.emb(img_list)
    def sample_save(self,model_path):
        with self.sess.graph.as_default():
            with self.sess.as_default():
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                tf.saved_model.simple_save(self.sess, model_path, inputs={'input:0': images_placeholder}, outputs={'embeddings:0': embeddings})
    





class EmbClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, emb_paths):
        self.name = name
        self.emb_paths = emb_paths
    def __str__(self):
        return self.name + ', ' + str(len(self.emb_paths)) + ' images'
    def __len__(self):
        return len(self.emb_paths)

class FaceClass():
    def __init__(self,aligned_pic,emb,faceid =None):
        self.alingned_pic = aligned_pic
        self.emb = emb;
        self.faceid = faceid
        if faceid is None:
            self.faceid = "F-"+str(uuid.uuid1())


class myclassify:
    def __init__(self,project_dir,classifier,bScale= False):
        self.tempFaceMap={}
        self.project_dir = project_dir
        self.data_dir = project_dir+"/data"
        self.temporay_data_dir=project_dir+"/tmpdata"
        self.embed_model_dir = project_dir+"/embed_model"
        self.classify_model_path = project_dir+"/classify.plk"
        # self.classify_train_model_path = project_dir+"/classify_train.plk"
        self.classify_mode=(classifier,None)
        self.dist_threshold = 0.89
        self.probab_threshold = 0.5
        self.isTraining = False
        self.bScale = bScale
        if not os.path.exists(self.project_dir):
            logging.error("project_dir={} is not exit".format(self.project_dir))
            sys.exit()
        if not os.path.exists(self.embed_model_dir):
            logging.error("embed_dir={} is not exit".format(self.embed_model_dir))
            sys.exit()
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.temporay_data_dir):
            os.makedirs(self.temporay_data_dir)

        self.facenet_ebeding_cls = facenet_ebeding(self.embed_model_dir)
        self.use_split_dataset = True
        self.min_nrof_images_per_class = 10
        self.nrof_train_images_per_class = 10
        self.batch_size = 1000
        self.image_size = 160


    def add_person(self,name):
        personid = "P-"+str(uuid.uuid1())
        os.makedirs(self.data_dir+"/"+personid)
        return personid
    def addFaceToPerson(self,personid,face):
        persondir = self.data_dir+"/"+personid
        facePicPath = persondir+"/"+face.faceid+".png"
        faceEmbPath = persondir+"/"+face.faceid+".emb"
        misc.imsave(facePicPath, face.alingned_pic)
        pickle.dump(face.emb,open(faceEmbPath,'wb'))
    def detectPaths(self,origin_picpaths_list,save_path=None):
        origin_pic_list=[]
        for i in range(len(origin_picpaths_list)):
            img = misc.imread(origin_picpaths_list[i])
            origin_pic_list.append(img)
        return self.detectImgs(origin_pic_list,save_path)

    def detectImgs(self,origin_pic_list,save_path=None):
        ret ={}
        box_truncate_list=[]
        box_list =[]
        detect = detectface()
        for i in range(len(origin_pic_list)):
            img = origin_pic_list[i]
            boxs = detect.detect_img(img)
            if boxs is None:
                logging.error("box is None")
                return
            for j in range(boxs.shape[0]):
                box_truncate = detect.truncate(img,boxs[j,0:4])
                box_truncate_list.append(box_truncate)
                box_list.append(boxs[j])
                if save_path:
                    img_save= save_path+"/"+str(round(time() * 1000))+"_"+str(j)+"_tmp.png"
                    logging.debug("img_save={}".format(img_save))
                    misc.imsave(img_save, box_truncate)
        logging.debug("len(box_truncate_list)={}".format(len(box_truncate_list)))
        if 0 == len(box_truncate_list):
            logging.error("box_truncate_list's is 0")
            return
        embs = self.facenet_ebeding_cls.embed(box_truncate_list)
        if embs is None:
            logging.error("embs is None")
        logging.debug("type(embs)={}".format(type(embs)))
        logging.debug("embs's shape={}".format(embs.shape))
        if len(box_truncate_list) != embs.shape[0]:
            logging.error("len(box_truncate_list) != embs.shape[0],{} {}".format(len(box_truncate_list),embs.shape[0]))
        for i in range(embs.shape[0]):
            face = FaceClass(box_truncate_list[i],embs[i,:])
            self.tempFaceMap[face.faceid]=face
            ret[face.faceid]=(box_list[i],face)
        return ret

    def distance(self,FaceList):
        faceNum = len(FaceList)
        retArr = np.full((faceNum, faceNum), 0,dtype=np.float) 
        distPrintStr=""
        for i in range(faceNum):
            for j in range(faceNum):
                if 0 == retArr[i][j]:
                    dist = np.sqrt(np.sum(np.square(np.subtract(FaceList[i].emb,FaceList[j].emb))))
                    retArr[i][j] = dist
                    retArr[j][i] = dist
        return retArr
    
    def iDentify(self,img):
        detectRet = self.detectImgs([img])


    def AutoProduceEmbs(self):
        emb_pics=[]
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            for filename in filenames:
                fullname = os.path.join(dirpath, filename)
                split_filename = os.path.splitext(fullname)
                filename_pre= split_filename[0]
                filename_suff =split_filename[1]
                logging.debug("fullname={},filename_pre={},filename_suff={}".format(fullname,filename_pre,filename_suff))
                if '.png' == filename_suff:
                    emb_filename = filename_pre+".emb"
                    if not os.path.exists(emb_filename):
                        emb_list =self.facenet_ebeding_cls.embed_paths([fullname])
                        if emb_list is not None:
                            logging.debug("write emb file={},shape={}".format(emb_filename,emb_list[0].shape))
                            pickle.dump(emb_list[0],open(emb_filename,'wb'))
                if '.emb' == filename_suff:
                    if not os.path.exists(filename_pre+'.png'):
                        os.remove(fullname)        
    def load_data(self):
        dataset = self.__get_emb_dataset()
        return self.__getEmbpathsLabelsPersonids(dataset)


    def load_data_bak(self):
        if self.use_split_dataset:
            dataset_tmp = facenet.get_dataset(self.data_dir)
            train_set, test_set = split_dataset(dataset_tmp, self.min_nrof_images_per_class, self.nrof_train_images_per_class)
            if (False == bClassify):
                dataset = train_set
            else:
                dataset = test_set
        else:
            dataset = facenet.get_dataset(self.data_dir)
        # Check that there are at least one training image per class
        dataset = [cls for cls in dataset  if len(cls.image_paths)>0]
        for cls in dataset:
            if 0 == len(cls.image_paths):
                logging.error("the personid={} picture num is 0".format(cls.name))
                return

        paths, labels = facenet.get_image_paths_and_labels(dataset)
        personids = [ cls.name for cls in dataset]
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        print('Loading feature extraction model')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(self.embed_model_dir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                print("batch_size=",self.batch_size," norf_images=",nrof_images," nrof_batches_per_epoch=",nrof_batches_per_epoch)
                for i in range(nrof_batches_per_epoch):
                    start_index = i*self.batch_size
                    end_index = min((i+1)*self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                return emb_array,labels,personids

    def fit_dir(self,bClassify = False):
        emb_list,labels,personids =self.load_data()
        max_emb_num = 3
        PersonInfoList=[]
        for i in range(len(personids)):
            item=(personids[i],[])
            PersonInfoList.append(item)
        
        for j in range(len(labels)):
            if max_emb_num > len(PersonInfoList[labels[j]][1]):
                PersonInfoList[labels[j]][1].append(emb_list[j])

        logging.debug("len(emb_list)={},len(labels)={},len(PersonInfoList)={},emb_list[0].shape={}".format(len(emb_list),len(labels),len(PersonInfoList),emb_list[0].shape))
        if (False == bClassify):
            # Train classifier
            self.isTraining = True
            print('Training classifier')
            X_train= np.array(emb_list,dtype=np.float32)
            if self.bScale:
                scaler = StandardScaler()
                self.fit = StandardScaler().fit(X_train)
                X_train = self.fit.transform(X_train)
            # X_train_scaled= X_train
            y_train = np.array(labels)

            model = self.classify_mode[0]
            # model = SVC(kernel='linear',probability=True)
            # model = SVC(kernel='linear')
            # model = SVC(kernel='line')
            # model = SVC(kernel='rbf')
            # model =LinearSVC(C=1,loss="hinge")

            
            # feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
            # model = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=len(personids),feature_columns=feature_columns)
            # model.fit(x=X_train, y=y_train, batch_size=1, steps=5000)


            # model = NeurosFaceClassifier(len(personids),200,1)
            model.fit(X_train,y_train)

    

            # model.fit(X_train_scaled, labels)
            # with open(self.classify_model_path, 'wb') as outfile:
                # pickle.dump((model, PersonInfoList), outfile)
            logging.debug("saved classifiler model to file={}".format(self.classify_model_path))
            self.classify_mode=(model,PersonInfoList)
        else:
            # Classify images
            print('Testing classifier')
            with open(self.classify_model_path, 'rb') as infile:
                (model, personids,PersonEmbDict) = pickle.load(infile)
            # logging.debug("load classifiler model to file={}".format(self.classify_model_path))

            predictions = model.predict_proba(emb_list)
            # print("predictions=",predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            # print("best_class_indices=",best_class_indices)
            logging.debug("len(best_class_indices)={}".format(len(best_class_indices)))
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            # print("best_class_probabilities=",best_class_probabilities)
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, personids[best_class_indices[i]], best_class_probabilities[i]))
            accuracy = np.mean(np.equal(best_class_indices, labels))
            print('Accuracy: %.3f' % accuracy)
    def predictProbaFaceid(self,faceid):
        logging.debug("faceid={}".format(faceid))
        if faceid not in self.tempFaceMap:
            logging.error("not find faceid={}".format(faceid))
            return
        return self.predictProbaEmbList([self.tempFaceMap[faceid].emb])

    def predictFaceid(self,faceid):
        logging.debug("faceid={}".format(faceid))
        if faceid not in self.tempFaceMap:
            logging.error("not find faceid={}".format(faceid))
            return
        ret = self.predictEmbList([self.tempFaceMap[faceid].emb])
        if 0 == len(ret):
            return None
        return ret[0] 

    def predictProbaEmbList(self,emb_list):
        logging.debug("len(emb_list)={}".format(len(emb_list)))
        ret_list=[]
        if self.classify_mode is None:
            logging.error("classify_mode is None")
            with open(self.classify_model_path, 'rb') as infile:
                self.classify_mode = pickle.load(infile)
        mode_tuple = self.classify_mode;
        model = mode_tuple[0]
        personids = mode_tuple[1]
        predictions = model.predict_proba(emb_list)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        for i in range(len(best_class_indices)):
            ret_list.append((personids[best_class_indices[i]],best_class_probabilities[i]))
            # print('%4d  %s: %.3f' % (i, personids[best_class_indices[i]], best_class_probabilities[i]))
        return ret_list

    def predictEmbList(self,emb_list):
        logging.debug("len(emb_list)={}".format(len(emb_list)))
        X_test = np.array(emb_list,dtype=np.float32)
        if self.bScale:
            X_test = self.fit.transform(X_test)
        ret_list=[]
        if self.classify_mode is None:
            if False == os.path.exists(self.classify_model_path):
                logging.error("no exit modelfile={}".format(self.classify_model_path))
                return []
            with open(self.classify_model_path, 'rb') as infile:
                self.classify_mode = pickle.load(infile)
        mode_tuple = self.classify_mode;
        model = mode_tuple[0]
        PersonInfoList = mode_tuple[1]
        # logging.debug("PersonInfoList={}".format(PersonInfoList))
        # best_class_indices = list(model.predict(X_test_scaled))
        best_class_indices,best_class_prob = list(model.predict_proba(X_test))
        logging.debug("best_class_indices={},best_class_prob={}".format(best_class_indices,best_class_prob))
        for i in range(len(best_class_indices)):
            personid = PersonInfoList[best_class_indices[i]][0]
            personid_probab = best_class_prob[i]
            dists = [np.sqrt(np.sum(np.square(np.subtract(item,emb_list[i])))) for item in PersonInfoList[best_class_indices[i]][1]]
            logging.debug("dists={}".format(dists))
            # dist = np.sqrt(np.sum(np.square(np.subtract(PersonInfoList[best_class_indices[i]][1][0],emb_list[i]))))
            dist = np.mean(dists)
            logging.debug("personid={},probab={},dist={}".format(personid,personid_probab,dist))
            if dist > self.dist_threshold:
                logging.debug("drop,personid={},dist={}".format(personid,dist))
                ret_list.append(None)
                continue
            if personid_probab < self.probab_threshold:
                logging.debug("drop,personid={},probab={}".format(personid,personid_probab))
                ret_list.append(None)
                continue
            
            logging.debug("get,personid={},dist={}".format(personid,dist))
            ret_list.append((personid,personid_probab,dist))
            # print('%4d  %s: %.3f' % (i, personids[best_class_indices[i]], best_class_probabilities[i]))
        return ret_list
    




    def fit_dir_bak(self,bClassify):
        if self.use_split_dataset:
            dataset_tmp = facenet.get_dataset(self.data_dir)
            train_set, test_set = split_dataset(dataset_tmp, self.min_nrof_images_per_class, self.nrof_train_images_per_class)
            if (False == bClassify):
                dataset = train_set
            else:
                dataset = test_set
        else:
            dataset = facenet.get_dataset(self.data_dir)
        # Check that there are at least one training image per class
        dataset = [cls for cls in dataset  if len(cls.image_paths)>0]
        for cls in dataset:
            if 0 == len(cls.image_paths):
                logging.error("the personid={} picture num is 0".format(cls.name))
                return

        paths, labels = facenet.get_image_paths_and_labels(dataset)
        
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        print('Loading feature extraction model')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(self.embed_model_dir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                print("batch_size=",self.batch_size," norf_images=",nrof_images," nrof_batches_per_epoch=",nrof_batches_per_epoch)
                for i in range(nrof_batches_per_epoch):
                    start_index = i*self.batch_size
                    end_index = min((i+1)*self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                
                classifier_filename_exp = os.path.expanduser(self.classify_model_path)
                if (False == bClassify):
                    # Train classifier
                    print('Training classifier')
                    model = SVC(kernel='linear', probability=True)
                    model.fit(emb_array, labels)
                    # Create a list of class names
                    class_names = [ cls.name.replace('_', ' ') for cls in dataset]
                    # Saving classifier model
                    with open(classifier_filename_exp, 'wb') as outfile:
                        pickle.dump((model, class_names), outfile)
                    print('Saved classifier model to file "%s"' % classifier_filename_exp)
                    
                else:
                    # Classify images
                    print('Testing classifier')
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
                    print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                    predictions = model.predict_proba(emb_array)
                    print("predictions=",predictions)
                    best_class_indices = np.argmax(predictions, axis=1)
                    print("best_class_indices=",best_class_indices)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    print("best_class_probabilities=",best_class_probabilities)
                    for i in range(len(best_class_indices)):
                        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    accuracy = np.mean(np.equal(best_class_indices, labels))
                    print('Accuracy: %.3f' % accuracy) 
    def __get_emb_dataset(self):
        path = self.data_dir
        dataset = []
        path_exp = os.path.expanduser(path)
        logging.debug("path_exp={}".format(path_exp))
        classes = [path for path in os.listdir(path_exp) \
                        if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            logging.debug("begin personid={},facedir={}".format(class_name,facedir))
            emb_paths = self.__get_emb_paths(facedir)
            dataset.append(EmbClass(class_name, emb_paths))
        return dataset

    def __getEmbpathsLabelsPersonids(self,dataset):
        emb_list = []
        labels_list = []
        personids =[]
        logging.debug("len(dataset)={}".format(len(dataset)))
        for i in range(len(dataset)):
            for j in range(len(dataset[i].emb_paths)):
                with open(dataset[i].emb_paths[j], 'rb') as infile:
                    emb = pickle.load(infile)
                    emb_list.append(emb)
            labels_list += [i] * len(dataset[i].emb_paths)
            personids.append(dataset[i].name)
        return emb_list, labels_list,personids   

    def __get_emb_paths(self,facedir):
        emb_paths = []
        if os.path.isdir(facedir):
            files = os.listdir(facedir)
            emb_paths = [os.path.join(facedir,file) for file in files if file.endswith('.emb')]
        return emb_paths


                

def train(args):
      
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # print("labels=",labels)
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            print("batch_size=",args.batch_size," norf_images=",nrof_images," nrof_batches_per_epoch=",nrof_batches_per_epoch)
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                print("predictions=",predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                print("best_class_indices=",best_class_indices)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                print("best_class_probabilities=",best_class_probabilities)
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set






class myclassify2:
    def __init__(self,project_dir,classifier,bScale= False):
        self.tempFaceMap={}
        self.project_dir = project_dir
        self.data_dir = project_dir+"/data"
        self.temporay_data_dir=project_dir+"/tmpdata"
        self.embed_model_dir = project_dir+"/embed_model"
        self.classify_model_path = project_dir+"/classify.plk"
        # self.classify_train_model_path = project_dir+"/classify_train.plk"
        self.classify_mode=(classifier,None)
        self.dist_threshold = 0.8
        self.probab_threshold = 0.5
        self.isTraining = False
        self.bScale = bScale
        if not os.path.exists(self.project_dir):
            logging.error("project_dir={} is not exit".format(self.project_dir))
            sys.exit()
        if not os.path.exists(self.embed_model_dir):
            logging.error("embed_dir={} is not exit".format(self.embed_model_dir))
            sys.exit()
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.temporay_data_dir):
            os.makedirs(self.temporay_data_dir)

        self.facenet_ebeding_cls = facenet_ebeding(self.embed_model_dir)
        self.use_split_dataset = True
        self.min_nrof_images_per_class = 10
        self.nrof_train_images_per_class = 10
        self.batch_size = 1000
        self.image_size = 160
        self.detect = detectface()


    def add_person(self,name):
        personid = "P-"+str(uuid.uuid1())
        os.makedirs(self.data_dir+"/"+personid)
        return personid
    def addFaceToPerson(self,personid,face):
        persondir = self.data_dir+"/"+personid
        facePicPath = persondir+"/"+face.faceid+".png"
        faceEmbPath = persondir+"/"+face.faceid+".emb"
        misc.imsave(facePicPath, face.alingned_pic)
        pickle.dump(face.emb,open(faceEmbPath,'wb'))
    def agumentPic(self,origin_pic):
        pass
    def detectPaths(self,origin_picpaths_list,save_path=""):
        origin_pic_list=[]
        for i in range(len(origin_picpaths_list)):
            img = misc.imread(origin_picpaths_list[i])
            origin_pic_list.append(img)
        return self.detectImgs(origin_pic_list,save_path)

    def detectImgs(self,origin_pic_list,save_path=""):
        logging.debug("enter")
        ret ={}
        box_truncate_list=[]
        box_list =[]
        for i in range(len(origin_pic_list)):
            img = origin_pic_list[i]
            boxs = self.detect.detect_img(img)
            if boxs is None:
                logging.error("box is None")
                return
            for j in range(boxs.shape[0]):
                box_truncate = self.detect.truncate(img,boxs[j,0:4])
                box_truncate_list.append(box_truncate)
                box_list.append(boxs[j])
                if len(save_path):
                    img_save= save_path+"/"+str(round(time() * 1000))+"_"+str(j)+"_tmp.png"
                    logging.debug("img_save={}".format(img_save))
                    misc.imsave(img_save, box_truncate)
        logging.debug("len(box_truncate_list)={}".format(len(box_truncate_list)))
        if 0 == len(box_truncate_list):
            logging.error("exit,box_truncate_list's is 0")
            return
        embs = self.facenet_ebeding_cls.embed(box_truncate_list)
        if embs is None:
            logging.error("embs is None")
        logging.debug("type(embs)={}".format(type(embs)))
        logging.debug("embs's shape={}".format(embs.shape))
        if len(box_truncate_list) != embs.shape[0]:
            logging.error("len(box_truncate_list) != embs.shape[0],{} {}".format(len(box_truncate_list),embs.shape[0]))
        for i in range(embs.shape[0]):
            face = FaceClass(box_truncate_list[i],embs[i,:])
            self.tempFaceMap[face.faceid]=face
            ret[face.faceid]=(box_list[i],face)
        logging.debug("have detect face,exit")
        return ret

    

    def distance(self,FaceList):
        faceNum = len(FaceList)
        retArr = np.full((faceNum, faceNum), 0,dtype=np.float) 
        distPrintStr=""
        for i in range(faceNum):
            for j in range(faceNum):
                if 0 == retArr[i][j]:
                    dist = np.sqrt(np.sum(np.square(np.subtract(FaceList[i].emb,FaceList[j].emb))))
                    retArr[i][j] = dist
                    retArr[j][i] = dist
        return retArr
    
    def iDentify(self,img):
        detectRet = self.detectImgs([img])


    def AutoProduceEmbs(self):
        emb_pics=[]
        for dirpath, dirnames, filenames in os.walk(self.data_dir):
            for filename in filenames:
                fullname = os.path.join(dirpath, filename)
                split_filename = os.path.splitext(fullname)
                filename_pre= split_filename[0]
                filename_suff =split_filename[1]
                logging.debug("fullname={},filename_pre={},filename_suff={}".format(fullname,filename_pre,filename_suff))
                if '.png' == filename_suff:
                    emb_filename = filename_pre+".emb"
                    if not os.path.exists(emb_filename):
                        emb_list =self.facenet_ebeding_cls.embed_paths([fullname])
                        if emb_list is not None:
                            logging.debug("write emb file={},shape={}".format(emb_filename,emb_list[0].shape))
                            pickle.dump(emb_list[0],open(emb_filename,'wb'))
                if '.emb' == filename_suff:
                    if not os.path.exists(filename_pre+'.png'):
                        os.remove(fullname)        
    def load_data(self):
        dataset = self.__get_emb_dataset()
        return self.__getEmbpathsLabelsPersonids(dataset)


    def load_data_bak(self):
        if self.use_split_dataset:
            dataset_tmp = facenet.get_dataset(self.data_dir)
            train_set, test_set = split_dataset(dataset_tmp, self.min_nrof_images_per_class, self.nrof_train_images_per_class)
            if (False == bClassify):
                dataset = train_set
            else:
                dataset = test_set
        else:
            dataset = facenet.get_dataset(self.data_dir)
        # Check that there are at least one training image per class
        dataset = [cls for cls in dataset  if len(cls.image_paths)>0]
        for cls in dataset:
            if 0 == len(cls.image_paths):
                logging.error("the personid={} picture num is 0".format(cls.name))
                return

        paths, labels = facenet.get_image_paths_and_labels(dataset)
        personids = [ cls.name for cls in dataset]
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        print('Loading feature extraction model')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(self.embed_model_dir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                print("batch_size=",self.batch_size," norf_images=",nrof_images," nrof_batches_per_epoch=",nrof_batches_per_epoch)
                for i in range(nrof_batches_per_epoch):
                    start_index = i*self.batch_size
                    end_index = min((i+1)*self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                return emb_array,labels,personids

    def fit_dir(self,bClassify = False):
        emb_list,labels,personids =self.load_data()
        max_emb_num = 5
        PersonInfoList=[]
        for i in range(len(personids)):
            item=(personids[i],[])
            PersonInfoList.append(item)
        
        for j in range(len(labels)):
            if max_emb_num > len(PersonInfoList[labels[j]][1]):
                PersonInfoList[labels[j]][1].append(emb_list[j])

        logging.debug("len(emb_list)={},len(labels)={},len(PersonInfoList)={},emb_list[0].shape={}".format(len(emb_list),len(labels),len(PersonInfoList),emb_list[0].shape))
        if (False == bClassify):
            # Train classifier
            self.isTraining = True
            print('Training classifier')
            X_train= np.array(emb_list,dtype=np.float32)
            if self.bScale:
                scaler = StandardScaler()
                self.fit = StandardScaler().fit(X_train)
                X_train = self.fit.transform(X_train)
            # X_train_scaled= X_train
            y_train = np.array(labels)

            model = self.classify_mode[0]
            # model = SVC(kernel='linear',probability=True)
            # model = SVC(kernel='linear')
            # model = SVC(kernel='line')
            # model = SVC(kernel='rbf')
            # model =LinearSVC(C=1,loss="hinge")

            
            # feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
            # model = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=len(personids),feature_columns=feature_columns)
            # model.fit(x=X_train, y=y_train, batch_size=1, steps=5000)


            # model = NeurosFaceClassifier(len(personids),200,1)
            model.fit(X_train,y_train)

    

            # model.fit(X_train_scaled, labels)
            # with open(self.classify_model_path, 'wb') as outfile:
                # pickle.dump((model, PersonInfoList), outfile)
            logging.debug("saved classifiler model to file={}".format(self.classify_model_path))
            self.classify_mode=(model,PersonInfoList)
        else:
            # Classify images
            print('Testing classifier')
            with open(self.classify_model_path, 'rb') as infile:
                (model, personids,PersonEmbDict) = pickle.load(infile)
            # logging.debug("load classifiler model to file={}".format(self.classify_model_path))

            predictions = model.predict_proba(emb_list)
            # print("predictions=",predictions)
            best_class_indices = np.argmax(predictions, axis=1)
            # print("best_class_indices=",best_class_indices)
            logging.debug("len(best_class_indices)={}".format(len(best_class_indices)))
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            # print("best_class_probabilities=",best_class_probabilities)
            for i in range(len(best_class_indices)):
                print('%4d  %s: %.3f' % (i, personids[best_class_indices[i]], best_class_probabilities[i]))
            accuracy = np.mean(np.equal(best_class_indices, labels))
            print('Accuracy: %.3f' % accuracy)
    def predictProbaFaceid(self,faceid):
        logging.debug("faceid={}".format(faceid))
        if faceid not in self.tempFaceMap:
            logging.error("not find faceid={}".format(faceid))
            return
        return self.predictProbaEmbList([self.tempFaceMap[faceid].emb])

    def predictFaceid(self,faceid):
        logging.debug("faceid={}".format(faceid))
        if faceid not in self.tempFaceMap:
            logging.error("not find faceid={}".format(faceid))
            return
        ret = self.predictEmbList([self.tempFaceMap[faceid].emb])
        if 0 == len(ret):
            return None
        return ret[0] 

    def predictProbaEmbList(self,emb_list):
        logging.debug("len(emb_list)={}".format(len(emb_list)))
        ret_list=[]
        if self.classify_mode is None:
            logging.error("classify_mode is None")
            with open(self.classify_model_path, 'rb') as infile:
                self.classify_mode = pickle.load(infile)
        mode_tuple = self.classify_mode;
        model = mode_tuple[0]
        personids = mode_tuple[1]
        predictions = model.predict_proba(emb_list)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        for i in range(len(best_class_indices)):
            ret_list.append((personids[best_class_indices[i]],best_class_probabilities[i]))
            # print('%4d  %s: %.3f' % (i, personids[best_class_indices[i]], best_class_probabilities[i]))
        return ret_list

    def predictEmbList(self,emb_list):
        logging.debug("len(emb_list)={}".format(len(emb_list)))
        X_test = np.array(emb_list,dtype=np.float32)
        if self.bScale:
            X_test = self.fit.transform(X_test)
        ret_list=[]
        if self.classify_mode is None:
            if False == os.path.exists(self.classify_model_path):
                logging.error("no exit modelfile={}".format(self.classify_model_path))
                return []
            with open(self.classify_model_path, 'rb') as infile:
                self.classify_mode = pickle.load(infile)
        mode_tuple = self.classify_mode;
        model = mode_tuple[0]
        PersonInfoList = mode_tuple[1]
        # logging.debug("PersonInfoList={}".format(PersonInfoList))
        # best_class_indices = list(model.predict(X_test_scaled))
        # best_class_indices,best_class_prob = list(model.predict_proba(X_test))
        probability =model.predict_proba(X_test)
        logging.debug("type(predict_test)={}".format(type(probability)))
        y_pred =np.argmax(probability,axis=1)
        y_probability =probability[:,y_pred].reshape((-1,))
        logging.debug("y_pred={},y_probabiltity={}".format(y_pred,y_probability))
        best_class_indices = y_pred
        best_class_prob = y_probability


        logging.debug("best_class_indices={},best_class_prob={}".format(best_class_indices,best_class_prob))
        for i in range(len(best_class_indices)):
            personid = PersonInfoList[best_class_indices[i]][0]
            personid_probab = best_class_prob[i]
            dists = [np.sqrt(np.sum(np.square(np.subtract(item,emb_list[i])))) for item in PersonInfoList[best_class_indices[i]][1]]
            logging.debug("dists={}".format(dists))
            # dist = np.sqrt(np.sum(np.square(np.subtract(PersonInfoList[best_class_indices[i]][1][0],emb_list[i]))))
            dist = np.mean(dists)
            logging.debug("personid={},probab={},dist={}".format(personid,personid_probab,dist))
            if dist > self.dist_threshold:
                logging.debug("drop,personid={},dist={}".format(personid,dist))
                ret_list.append(None)
                continue
            if personid_probab < self.probab_threshold:
                logging.debug("drop,personid={},probab={}".format(personid,personid_probab))
                ret_list.append(None)
                continue
            
            logging.debug("get,personid={},dist={},probab={}".format(personid,dist,personid_probab))
            ret_list.append((personid,personid_probab,dist))
            # print('%4d  %s: %.3f' % (i, personids[best_class_indices[i]], best_class_probabilities[i]))
        return ret_list
    




    def fit_dir_bak(self,bClassify):
        if self.use_split_dataset:
            dataset_tmp = facenet.get_dataset(self.data_dir)
            train_set, test_set = split_dataset(dataset_tmp, self.min_nrof_images_per_class, self.nrof_train_images_per_class)
            if (False == bClassify):
                dataset = train_set
            else:
                dataset = test_set
        else:
            dataset = facenet.get_dataset(self.data_dir)
        # Check that there are at least one training image per class
        dataset = [cls for cls in dataset  if len(cls.image_paths)>0]
        for cls in dataset:
            if 0 == len(cls.image_paths):
                logging.error("the personid={} picture num is 0".format(cls.name))
                return

        paths, labels = facenet.get_image_paths_and_labels(dataset)
        
        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        print('Loading feature extraction model')
        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model(self.embed_model_dir)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                print('Calculating features for images')
                nrof_images = len(paths)
                nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))
                emb_array = np.zeros((nrof_images, embedding_size))
                print("batch_size=",self.batch_size," norf_images=",nrof_images," nrof_batches_per_epoch=",nrof_batches_per_epoch)
                for i in range(nrof_batches_per_epoch):
                    start_index = i*self.batch_size
                    end_index = min((i+1)*self.batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, self.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                
                classifier_filename_exp = os.path.expanduser(self.classify_model_path)
                if (False == bClassify):
                    # Train classifier
                    print('Training classifier')
                    model = SVC(kernel='linear', probability=True)
                    model.fit(emb_array, labels)
                    # Create a list of class names
                    class_names = [ cls.name.replace('_', ' ') for cls in dataset]
                    # Saving classifier model
                    with open(classifier_filename_exp, 'wb') as outfile:
                        pickle.dump((model, class_names), outfile)
                    print('Saved classifier model to file "%s"' % classifier_filename_exp)
                    
                else:
                    # Classify images
                    print('Testing classifier')
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names) = pickle.load(infile)
                    print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                    predictions = model.predict_proba(emb_array)
                    print("predictions=",predictions)
                    best_class_indices = np.argmax(predictions, axis=1)
                    print("best_class_indices=",best_class_indices)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    print("best_class_probabilities=",best_class_probabilities)
                    for i in range(len(best_class_indices)):
                        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    accuracy = np.mean(np.equal(best_class_indices, labels))
                    print('Accuracy: %.3f' % accuracy) 
    def __get_emb_dataset(self):
        path = self.data_dir
        dataset = []
        path_exp = os.path.expanduser(path)
        logging.debug("path_exp={}".format(path_exp))
        classes = [path for path in os.listdir(path_exp) \
                        if os.path.isdir(os.path.join(path_exp, path))]
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            logging.debug("begin personid={},facedir={}".format(class_name,facedir))
            emb_paths = self.__get_emb_paths(facedir)
            dataset.append(EmbClass(class_name, emb_paths))
        return dataset

    def __getEmbpathsLabelsPersonids(self,dataset):
        emb_list = []
        labels_list = []
        personids =[]
        logging.debug("len(dataset)={}".format(len(dataset)))
        for i in range(len(dataset)):
            for j in range(len(dataset[i].emb_paths)):
                with open(dataset[i].emb_paths[j], 'rb') as infile:
                    emb = pickle.load(infile)
                    emb_list.append(emb)
            labels_list += [i] * len(dataset[i].emb_paths)
            personids.append(dataset[i].name)
        return emb_list, labels_list,personids   

    def __get_emb_paths(self,facedir):
        emb_paths = []
        if os.path.isdir(facedir):
            files = os.listdir(facedir)
            emb_paths = [os.path.join(facedir,file) for file in files if file.endswith('.emb')]
        return emb_paths



class TvuFace:
    def __init__(self,projectdir,datadir,classifier,bScale= False):
        logging.info("TvuFace init begin")
        self.projectdir = projectdir
        self.bScale = bScale
        self.datadir = datadir
        self.classifier = classifier

        self.dist_threshold = 0.5
        self.probab_threshold =0.5
        self.minBoxArea = 6400
        self.detect = detectface()
        self.detectNum =0
        self.detectSumTime =0
        self.detectMaxTime =0

        self.lock = threading.Lock()
        threading.Thread(target=TvuFace._delay_init,args=(self,)).start()
        logging.info("TvuFace init end")
    def _delay_init(self):
        with self.lock:
            logging.info("delay init begin")
            self.database = LocalDataManger(self.datadir,".png",".emb",isSyncSrcx=True)
            self.database.clear(512,isRemove=True)
            getdata = self.database.GetBatchData()
            self.embX= getdata[0]
            self.y= getdata[1]
            self.srcX_list = getdata[2]
            self.xIds_list = getdata[3]
            self.yIds_list = getdata[4]
            self.fit()
            self.facenet_ebeding_cls = facenet_ebeding(self.projectdir+"/embed_model")
            logging.info("delay init end")
    def clear(self,classifier):
        with self.lock:
            self.classifier = classifier
            self.database.clear(512)
            self.fit()
            return
    def detectPaths(self,id,origin_picpaths_list,save_path=""):
        origin_pic_list=[]
        for i in range(len(origin_picpaths_list)):
            img = misc.imread(origin_picpaths_list[i])
            origin_pic_list.append(img)
        return self.detectImg(id,origin_pic_list,save_path)

    def detectImgAndEmbed(self,id,origin_pic,faceMinisize=0,save_path=""):
        logging.debug("enter,detect_img begin")
        ret ={}
        box_truncate_list=[]
        box_list =[]    
        boxs = self.detect.detect_img(origin_pic,faceMinisize)
        if boxs is None:
            logging.error("box is None")
            return
        for j in range(boxs.shape[0]):
            BoxWidth = int(boxs[j][2] -boxs[j][0])
            BoxHeight = int(boxs[j][3] -boxs[j][1])
            if BoxWidth * BoxHeight < self.minBoxArea:
                logging.info("area is too small,id={},boxwidth={},boxheight={}".format(id,BoxWidth,BoxHeight))
                continue

            # if BoxHeight - BoxWidth > 0:
            #     boxs[j][1] = boxs[j][1]+(BoxHeight - BoxWidth)

            box_truncate = self.detect.truncate(origin_pic,boxs[j,0:4])
            box_truncate_list.append(box_truncate)
            box_list.append(boxs[j])
            if len(save_path):
                img_save= save_path+"/"+str(round(time() * 1000))+"_"+str(j)+"_tmp.png"
                logging.debug("img_save={}".format(img_save))
                misc.imsave(img_save, box_truncate)
        logging.debug("detect_img end,len(box_truncate_list)={}".format(len(box_truncate_list)))
        if 0 == len(box_truncate_list):
            logging.error("exit,box_truncate_list's is 0")
            return
        embs = self.facenet_ebeding_cls.embed(box_truncate_list)
        if embs is None:
            logging.error("embs is None")
        logging.debug("embs's shape={}".format(embs.shape))
        if len(box_truncate_list) != embs.shape[0]:
            logging.error("len(box_truncate_list) != embs.shape[0],{} {}".format(len(box_truncate_list),embs.shape[0]))
        for i in range(embs.shape[0]):
            left = int(box_list[i][0])
            top = int(box_list[i][1])
            width = int(box_list[i][2] -box_list[i][0])
            height = int(box_list[i][3] -box_list[i][1])
            faceid ="F-"+id + "-"+str(left)+"-"+str(top)+"-"+str(width)+"-"+str(height)
            face = FaceClass(box_truncate_list[i],embs[i,:],faceid)
            ret[face.faceid]=([left,top,width,height],face)
            logging.info("detect facenum={},one emb faceid={}".format(embs.shape[0],faceid))
        logging.debug("have detect face,exit")
        return ret
    

    def detectImg(self,id,origin_pic,faceMinisize=0,save_path=""):
        logging.debug("enter")
        ret ={}
        box_truncate_list=[]
        box_list =[] 
        detectBeginTime = int(round(time() * 1000))   
        boxs = self.detect.detect_img(origin_pic,faceMinisize)
        detectEndTime = int(round(time() * 1000))
        costTime = detectEndTime - detectBeginTime
        self.detectNum += 1
        self.detectSumTime += costTime
        if costTime > self.detectMaxTime:
            self.detectMaxTime = costTime

        logging.info("detect performance,id={},costTime={},maxTime={},detectNum={},averageTime={}".format(id,costTime,self.detectMaxTime,self.detectNum,self.detectSumTime/self.detectNum))

        if boxs is None:
            logging.error("box is None")
            return
        for j in range(boxs.shape[0]):
            BoxWidth = int(boxs[j][2] -boxs[j][0])
            BoxHeight = int(boxs[j][3] -boxs[j][1])
            if BoxWidth * BoxHeight < self.minBoxArea:
                logging.info("area is too small,id={},boxwidth={},boxheight={}".format(id,BoxWidth,BoxHeight))
                continue
            # if BoxHeight - BoxWidth > 0:
            #     boxs[j][1] = boxs[j][1]+(BoxHeight - BoxWidth)
            left = int(boxs[j][0])
            top = int(boxs[j][1])
            width = int(boxs[j][2] -boxs[j][0])
            height = int(boxs[j][3] -boxs[j][1])
            faceid ="F-"+id + "-"+str(left)+"-"+str(top)+"-"+str(width)+"-"+str(height)
            face = FaceClass(None,None,faceid)
            ret[faceid]=([left,top,width,height],face)
            logging.info("detect facenum={},one faceid={}".format(boxs.shape[0],faceid))
        logging.info("exit,id={},len(ret)={},boxs.shape={}".format(id,len(ret),boxs.shape))
        return ret

    def detectAlinedImgs(self,list_id,list_alinedImag):
        ret ={}
        embs = self.facenet_ebeding_cls.embed(list_alinedImag)
        if embs is None:
            logging.error("embs is None")
            return
        logging.debug("embs's shape={}".format(embs.shape))
        if len(list_alinedImag) != embs.shape[0]:
            logging.error("len(list_alinedImag)={},embs.shape[0]={}".format(len(list_alinedImag),embs.shape[0]))
            return
        for i in range(embs.shape[0]):
            faceid = list_id[i]
            face = FaceClass(list_alinedImag[i],embs[i,:],faceid)
            ret[face.faceid]=([0,0,0,0],face)
        logging.debug("have detect face,exit")
        return ret         
    def fit(self):
        getdata = self.database.GetBatchData()
        self.embX= getdata[0]
        self.y= getdata[1]
        self.srcX_list = getdata[2]
        self.xIds_list = getdata[3]
        self.yIds_list = getdata[4]
        logging.debug("yIds={}".format(self.yIds_list))

        self.classifier.fit(self.embX,self.y)
        return
    def predict(self,emb_list):
        ret_list=[]
        model = self.classifier
        logging.debug("len(emb_list)={}".format(len(emb_list)))
        X_test = np.array(emb_list,dtype=np.float32)
        probability =model.predict_proba(X_test)
        logging.debug("type(predict_test)={}".format(type(probability)))
        y_pred =np.argmax(probability,axis=1)
        y_probability =probability[:,y_pred].reshape((-1,))
        logging.debug("y_pred={},y_probabiltity={}".format(y_pred,y_probability))
        best_class_indices = y_pred
        best_class_prob = y_probability
        logging.debug("best_class_indices={},best_class_prob={}".format(best_class_indices,best_class_prob))
        for i in range(len(best_class_indices)):
            personid = self.yIds_list[best_class_indices[i]]
            personid_probab = best_class_prob[i]
            personClass = self.database.GetClass(personid)
            if personClass is None:
                logging.error("personClass is None,personid={}".format(personid))
                continue
            srcAndEmbs = personClass.items.values()
            for OnesrcAndEmb in srcAndEmbs:
                logging.debug("shape(OnesrcAndEmb[1])={}".format(OnesrcAndEmb[1].shape))
            
            
            get_distance = 100
            isSamePerson = False
            for OnesrcAndEmb in srcAndEmbs:
                get_distance =np.sqrt(np.sum(np.square(np.subtract(OnesrcAndEmb[1],emb_list[i]))))
                if get_distance < self.dist_threshold:
                    isSamePerson = True
                    break
                # if  get_distance < min_distance:
                #     min_distance = get_distance

            # dists = [np.sqrt(np.sum(np.square(np.subtract(OnesrcAndEmb[1],emb_list[i])))) for OnesrcAndEmb in srcAndEmbs]
            # logging.debug("dists={}".format(dists))
            # dist = np.mean(dists)
            logging.debug("personid={},probab={},get_distance={}".format(personid,personid_probab,get_distance))
            if False == isSamePerson:
                logging.debug("drop,personid={},get_distance={}".format(personid,get_distance))
                ret_list.append(None)
                continue
            if personid_probab < self.probab_threshold:
                logging.debug("drop,personid={},probab={}".format(personid,personid_probab))
                ret_list.append(None)
                continue
            logging.debug("get,personid={},get_distance={},probab={}".format(personid,get_distance,personid_probab))
            ret_list.append((personid,personid_probab,get_distance))
            # print('%4d  %s: %.3f' % (i, personids[best_class_indices[i]], best_class_probabilities[i]))
        return ret_list        
    def addFaceToPerson(self,personid,faceid,srcX,embX):
        self.database.AddData(personid,faceid,srcX,embX)
        return




def train(args):
      
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            # print("labels=",labels)
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            print("batch_size=",args.batch_size," norf_images=",nrof_images," nrof_batches_per_epoch=",nrof_batches_per_epoch)
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                print("predictions=",predictions)
                best_class_indices = np.argmax(predictions, axis=1)
                print("best_class_indices=",best_class_indices)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                print("best_class_probabilities=",best_class_probabilities)
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                





def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set





def main(args):
    sleep(random.random())
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, output_dir, ' '.join(sys.argv))
    print("args.input_dir=",args.input_dir," src_path=",src_path," sys.argv=",sys.argv)
    dataset = facenet.get_dataset(args.input_dir)

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
    factor = 0.709 # scale factor

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)

    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if args.random_order:
            random.shuffle(dataset)
        for cls in dataset:
            output_class_dir = os.path.join(output_dir, cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if args.random_order:
                    random.shuffle(cls.image_paths)
            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                print(image_path)
                if not os.path.exists(output_filename):
                    try:
                        img = misc.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        print("imgfile:",image_path," bouding_boxes'sshape=",bounding_boxes.shape," bounding_boxes=",bounding_boxes)
                        nrof_faces = bounding_boxes.shape[0]
                        if nrof_faces>0:
                            det = bounding_boxes[:,0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]
                            if nrof_faces>1:
                                if args.detect_multiple_faces:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                    det_arr.append(det[index,:])
                            else:
                                det_arr.append(np.squeeze(det))

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0]-args.margin/2, 0)
                                bb[1] = np.maximum(det[1]-args.margin/2, 0)
                                bb[2] = np.minimum(det[2]+args.margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+args.margin/2, img_size[0])
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                scaled = misc.imresize(cropped, (args.image_size, args.image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if args.detect_multiple_faces:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                misc.imsave(output_filename_n, scaled)
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir', type=str, help='Directory with unaligned images.')
    parser.add_argument('output_dir', type=str, help='Directory with aligned face thumbnails.')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
