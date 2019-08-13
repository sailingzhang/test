import os
import logging
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.contrib.layers import batch_norm

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'





class NeurosFaceClassifier_Bak2:
    def __init__(self,trainSteps=1000,bachSize=1):
        logging.debug("myfirstmode init,trainSteps={},bachSize={},bScale={}".format(trainSteps,bachSize,bScale))
        self.scaled_fit = None
        hidden1_n = 300
        hidden2_n = 100
        self.trainSteps = trainSteps
        self.bachSize = bachSize
        self.bInit = False
        self.sess = tf.Session()
        self.sess.as_default()
        self.test_var = tf.Variable([1,2,3])


        self.X = tf.placeholder(shape=(None,512),dtype=tf.float32)
        self.y = tf.placeholder(shape=(None),dtype=tf.int32)
        with tf.name_scope("constant_hidden"):
            hidden1 = fully_connected(self.X,hidden1_n,scope="hidden1")
            self.last_hidden = fully_connected(hidden1,hidden2_n,scope="hidden2")

    def fit(self,X,y):
        # output_n = len(np.unique(y, return_counts=False))
        output_n = 100
        logging.debug("begin fit,shape(X)={},shape(y)={},output_n={},bScale={}".format(X.shape,y.shape,output_n,self.bScale))
        learning_rate = 0.05
        saver = tf.train.Saver()
        with tf.name_scope("dynamic_output"):
            self.logits = fully_connected(self.last_hidden,output_n,activation_fn=None)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            self.loss = tf.reduce_mean(xentropy,name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.training_op = optimizer.minimize(self.loss)
    
        init = tf.global_variables_initializer()

        test_var_op = self.test_var.assign(self.test_var+1)
        with self.sess.as_default():
            logging.debug("begin train  session,trainSteps={},bachSize={}".format(self.trainSteps,self.bachSize))
            if False ==self.bInit:
                init.run()
            for epoch in range(self.trainSteps):
                rand_index = np.random.permutation(len(X))
                X_bachs = np.array_split(X[rand_index],self.bachSize)
                y_bachs = np.array_split(y[rand_index],self.bachSize)
                for xbach,ybach in  zip(X_bachs,y_bachs):
                    self.sess.run(self.training_op,feed_dict={self.X:xbach,self.y:ybach})
                loss = self.sess.run(self.loss,feed_dict={self.X:xbach,self.y:ybach})
                test_var =self.sess.run(test_var_op)
                logging.debug("epoch={},loss={},test_var={}".format(epoch,loss,test_var))
            # saver.save(self.sess,"./mymodel.ckpt")

    def predict(self,X):
        saver = tf.train.Saver()
        with self.sess.as_default():
            # saver.restore(self.sess,"./mymodel.ckpt")
            proability =self.sess.run(self.logits,feed_dict={self.X:X})
            y_pred =np.argmax(proability,axis=1)
            logging.debug("predict test_var={}".format(self.test_var.eval()))
            return y_pred





class NeurosFaceClassifier_Bak:
    def __init__(self,outNum,trainSteps=1000,bachSize=1,bScale=False):
        logging.debug("myfirstmode init,outNum={},trainSteps={},bachSize={},bScale={}".format(outNum,trainSteps,bachSize,bScale))
        self.scaled_fit = None
        hidden1_n = 300
        hidden2_n = 100
        output_n = outNum    
        self.trainSteps = trainSteps
        self.bachSize = bachSize
        self.bScale = bScale

        self.X = tf.placeholder(shape=(None,512),dtype=tf.float32)
        self.y = tf.placeholder(shape=(None),dtype=tf.int32)
        with tf.name_scope("facednn"):
            hidden1 = fully_connected(self.X,hidden1_n,scope="hidden1")
            hidden2 = fully_connected(hidden1,hidden2_n,scope="hidden2")
            logits = fully_connected(hidden2,output_n,scope="output",activation_fn=None)
        self.logits =logits

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            loss = tf.reduce_mean(xentropy,name="loss")
        self.loss = loss

    def fit(self,X,y):
        logging.debug("begin fit,shape(X)={},shape(y)={},bScale={}".format(X.shape,y.shape,self.bScale))
        learning_rate = 0.02
        if self.bScale:
            self.scaled_fit = StandardScaler().fit(X)
            X = self.scaled_fit.transform(X)
        saver = tf.train.Saver()
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.training_op = optimizer.minimize(self.loss)
    
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            logging.debug("begin train  session,trainSteps={},bachSize={}".format(self.trainSteps,self.bachSize))
            init.run()
            for epoch in range(self.trainSteps):
                rand_index = np.random.permutation(len(X))
                X_bachs = np.array_split(X[rand_index],self.bachSize)
                y_bachs = np.array_split(y[rand_index],self.bachSize)
                for xbach,ybach in  zip(X_bachs,y_bachs):
                    sess.run(self.training_op,feed_dict={self.X:xbach,self.y:ybach})
                loss = sess.run(self.loss,feed_dict={self.X:xbach,self.y:ybach})
                logging.debug("epoch={},loss={}".format(epoch,loss))
            saver.save(sess,"./mymodel.ckpt")

    def predict(self,X):
        logging.debug("begin predict")
        saver = tf.train.Saver()
        if self.bScale:
            X = self.scaled_fit.transform(X)
        with tf.Session() as sess:
            saver.restore(sess,"./mymodel.ckpt")
            proability = self.logits.eval(feed_dict={self.X:X})
            y_pred =np.argmax(proability,axis=1)
        return y_pred




class NeurosFaceClassifier_Bak3:
    def __init__(self,trainSteps=1000,bachSize=1,bScale=False):
        logging.debug("myfirstmode init,trainSteps={},bachSize={},bScale={}".format(trainSteps,bachSize,bScale))
        self.scaled_fit = None
        hidden1_n = 300
        hidden2_n = 200
        hidden3_n = 100
        self.output_n = 100
        self.trainSteps = trainSteps
        self.bachSize = bachSize
        self.bScale = bScale
        self.bInit = False
        self.sess = tf.Session()
        self.sess.as_default()
        self.test_var = tf.Variable([1,2,3])
        self.learning_rate = 0.05


        self.X = tf.placeholder(shape=(None,512),dtype=tf.float32)
        self.y = tf.placeholder(shape=(None),dtype=tf.int32)
        with tf.name_scope("constant_hidden"):
            hidden1 = fully_connected(self.X,hidden1_n,scope="hidden1")
            hidden2 = fully_connected(hidden1,hidden2_n,scope="hidden2")
            self.last_hidden = fully_connected(hidden2,hidden3_n,scope="hidden3")


        with tf.name_scope("dynamic_output"):
            self.logits = fully_connected(self.last_hidden,self.output_n,activation_fn=None)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            self.loss = tf.reduce_mean(xentropy,name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            self.training_op = optimizer.minimize(self.loss)
    
        self.test_var_op = self.test_var.assign(self.test_var+1)

    def fit(self,X,y):
        # output_n = len(np.unique(y, return_counts=False))
        # self.output_n = 100
        logging.debug("begin fit,shape(X)={},shape(y)={},output_n={},bScale={}".format(X.shape,y.shape,self.output_n,self.bScale))
        

        if self.bScale:
            self.scaled_fit = StandardScaler().fit(X)
            X = self.scaled_fit.transform(X)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        with self.sess.as_default():
            logging.debug("begin train  session,trainSteps={},bachSize={}".format(self.trainSteps,self.bachSize))
            if False ==self.bInit:
                init.run()
                self.bInit = True
            else:
                init.run()
                # pass
                # saver.restore(self.sess,"./mymodel.ckpt")
            for epoch in range(self.trainSteps):
                rand_index = np.random.permutation(len(X))
                X_bachs = np.array_split(X[rand_index],self.bachSize)
                y_bachs = np.array_split(y[rand_index],self.bachSize)
                for xbach,ybach in  zip(X_bachs,y_bachs):
                    self.sess.run(self.training_op,feed_dict={self.X:xbach,self.y:ybach})
                loss = self.sess.run(self.loss,feed_dict={self.X:xbach,self.y:ybach})
                test_var =self.sess.run(self.test_var_op)
                logging.debug("epoch={},loss={},test_var={},grapth={}".format(epoch,loss,test_var,tf.get_default_graph()))
            # saver.save(self.sess,"./mymodel.ckpt")

    def predict(self,X):
        saver = tf.train.Saver()
        if self.bScale:
            X = self.scaled_fit.transform(X)

        with self.sess.as_default():
            # saver.restore(self.sess,"./mymodel.ckpt")
            proability =self.sess.run(self.logits,feed_dict={self.X:X})
            y_pred =np.argmax(proability,axis=1)
            logging.debug("predict test_var={}".format(self.test_var.eval()))
            return y_pred




class NeurosFaceClassifier:
    def __init__(self,trainSteps=1000,bachSize=1,bScale=False):
        logging.debug("myfirstmode init,trainSteps={},bachSize={},bScale={}".format(trainSteps,bachSize,bScale))
        self.scaled_fit = None
        hidden1_n = 300
        hidden2_n = 100
        self.trainSteps = trainSteps
        self.bachSize = bachSize
        self.bScale = bScale
        self.bInit = False
        self.sess = tf.Session()
        self.sess.as_default()
        self.test_var = tf.Variable([1,2,3])


        self.X = tf.placeholder(shape=(None,512),dtype=tf.float32)
        self.y = tf.placeholder(shape=(None),dtype=tf.int32)
        self.is_training = tf.placeholder(shape=(),dtype=tf.bool)



        self.bn_params = {
        'is_training': self.is_training,
        'decay': 0.99,
        'updates_collections': None
        }

        with tf.name_scope("constant_hidden"):
            hidden1 = fully_connected(self.X,hidden1_n,scope="hidden1")
            self.last_hidden = fully_connected(hidden1,hidden2_n,scope="hidden2")

    def fit(self,X,y):
        output_n = len(np.unique(y, return_counts=False))
        # output_n = 100
        logging.debug("begin fit,shape(X)={},shape(y)={},output_n={},bScale={}".format(X.shape,y.shape,output_n,self.bScale))
        learning_rate = 0.05

        if self.bScale:
            self.scaled_fit = StandardScaler().fit(X)
            X = self.scaled_fit.transform(X)

        saver = tf.train.Saver()
        with tf.name_scope("dynamic_output"):
            self.logits = fully_connected(self.last_hidden,output_n,activation_fn=None)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            self.loss = tf.reduce_mean(xentropy,name="loss")

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.training_op = optimizer.minimize(self.loss)
    
        init = tf.global_variables_initializer()

        test_var_op = self.test_var.assign(self.test_var+1)
        with self.sess.as_default():
            logging.debug("begin train  session,trainSteps={},bachSize={}".format(self.trainSteps,self.bachSize))
            if False ==self.bInit:
                init.run()
            else:
                pass
            for epoch in range(self.trainSteps):
                rand_index = np.random.permutation(len(X))
                X_bachs = np.array_split(X[rand_index],self.bachSize)
                y_bachs = np.array_split(y[rand_index],self.bachSize)
                for xbach,ybach in  zip(X_bachs,y_bachs):
                    self.sess.run(self.training_op,feed_dict={self.X:xbach,self.y:ybach})
                loss = self.sess.run(self.loss,feed_dict={self.X:xbach,self.y:ybach})
                test_var =self.sess.run(test_var_op)
                logging.debug("epoch={},loss={},test_var={}".format(epoch,loss,test_var))
            # saver.save(self.sess,"./mymodel.ckpt")

    def predict(self,X):
        saver = tf.train.Saver()
        if self.bScale:
            X = self.scaled_fit.transform(X)

        with self.sess.as_default():
            # saver.restore(self.sess,"./mymodel.ckpt")
            proability =self.sess.run(self.logits,feed_dict={self.X:X})
            y_pred =np.argmax(proability,axis=1)
            logging.debug("predict test_var={}".format(self.test_var.eval()))
            return y_pred


    def predict_proba(self,X):
        logging.debug("len(X)={}".format(len(X)))
        saver = tf.train.Saver()
        if self.bScale:
            X = self.scaled_fit.transform(X)

        with self.sess.as_default():
            # saver.restore(self.sess,"./mymodel.ckpt")
            logits,probability =self.sess.run([self.logits,tf.nn.softmax(self.logits)],feed_dict={self.X:X})
            y_pred =np.argmax(probability,axis=1)
            logging.debug("logits={},probability={},y_pred={}".format(logits,probability,y_pred))
            y_probability =probability[:,y_pred].reshape((-1,))
            # logging.debug("predict test_var={}".format(self.test_var.eval()))
            logging.debug("logits={},probability={},y_pred={},y_probability={}".format(logits,probability,y_pred,y_probability))
            return y_pred,y_probability




class NeurosFaceClassifierStd:
    def __init__(self,trainSteps=1000,bachSize=1,bScale=False,thresholdScore=6):
        logging.debug("myfirstmode init,trainSteps={},bachSize={},bScale={}".format(trainSteps,bachSize,bScale))
        self.scaled_fit = None
        hidden1_n = 300
        hidden2_n = 100
        self.trainSteps = trainSteps
        self.bachSize = bachSize
        self.bScale = bScale
        self.bInit = False
        self.graph = tf.Graph()
        self.graph.as_default()
        self.sess = tf.Session(graph=self.graph)
        self.sess.as_default()
        self.thresholdScore = thresholdScore

        with self.graph.as_default():
            self.test_var = tf.Variable([1,2,3])
            self.X = tf.placeholder(shape=(None,512),dtype=tf.float32)
            self.y = tf.placeholder(shape=(None),dtype=tf.int32)
            self.is_training = tf.placeholder(shape=(),dtype=tf.bool)
            self.bn_params = {
            'is_training': self.is_training,
            'decay': 0.99,
            'updates_collections': None
            }

            with tf.name_scope("constant_hidden"):
                hidden1 = fully_connected(self.X,hidden1_n,scope="hidden1")
                self.last_hidden = fully_connected(hidden1,hidden2_n,scope="hidden2")

    def fit(self,X,y):
        output_n = len(np.unique(y, return_counts=False))
        # output_n = 100
        logging.debug("begin fit,id={},shape(X)={},shape(y)={},output_n={},bScale={}".format(id(self),X.shape,y.shape,output_n,self.bScale))
        learning_rate = 0.05

        if self.bScale:
            self.scaled_fit = StandardScaler().fit(X)
            X = self.scaled_fit.transform(X)

        with self.graph.as_default():
            # saver = tf.train.Saver()
            with tf.name_scope("dynamic_output"):
                self.logits = fully_connected(self.last_hidden,output_n,activation_fn=None)
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
                self.loss = tf.reduce_mean(xentropy,name="loss")

            with tf.name_scope("train"):
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
                self.training_op = optimizer.minimize(self.loss)
        
            init = tf.global_variables_initializer()

            test_var_op = self.test_var.assign(self.test_var+1)
            with self.sess.as_default():
                logging.debug("begin train,graph={},trainSteps={},bachSize={}".format(self.sess.graph,self.trainSteps,self.bachSize))
                logging.debug("init's graph={},default grapth={},self.graph={}".format(init.graph,tf.get_default_graph(),self.graph))
                if False ==self.bInit:
                    init.run()
                else:
                    pass
                for epoch in range(self.trainSteps):
                    rand_index = np.random.permutation(len(X))
                    X_bachs = np.array_split(X[rand_index],self.bachSize)
                    y_bachs = np.array_split(y[rand_index],self.bachSize)
                    for xbach,ybach in  zip(X_bachs,y_bachs):
                        self.sess.run(self.training_op,feed_dict={self.X:xbach,self.y:ybach})
                    loss = self.sess.run(self.loss,feed_dict={self.X:xbach,self.y:ybach})
                    test_var =self.sess.run(test_var_op)
                logging.debug("epoch={},loss={},test_var={}".format(epoch,loss,test_var))
                # saver.save(self.sess,"./mymodel.ckpt")

    def predict(self,X):
        saver = tf.train.Saver()
        if self.bScale:
            X = self.scaled_fit.transform(X)
        with self.graph.as_default():
            with self.sess.as_default():
                # saver.restore(self.sess,"./mymodel.ckpt")
                proability =self.sess.run(self.logits,feed_dict={self.X:X})
                y_pred =np.argmax(proability,axis=1)
                logging.debug("predict test_var={}".format(self.test_var.eval()))
                return y_pred


    def predict_proba(self,X):
        logging.debug("id={},len(X)={}".format(id(self),len(X)))
        # saver = tf.train.Saver()
        if self.bScale:
            X = self.scaled_fit.transform(X)
        with self.graph.as_default():
            with self.sess.as_default():
                # saver.restore(self.sess,"./mymodel.ckpt")
                logits,probability =self.sess.run([self.logits,tf.nn.softmax(self.logits)],feed_dict={self.X:X})
                logging.debug("probab={},logits={}".format(probability,logits))
                for i in range(len(logits)):
                    maxscore = max(logits[i])
                    if maxscore < self.thresholdScore:
                        logging.info("the maxscore={} is too small,drop".format(maxscore))
                        probability[i] = np.zeros(probability[i].shape)
                return probability
