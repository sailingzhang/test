import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append("..")

# import tensorflow as tf
# import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import fetch_mldata
# from datetime import datetime
import numpy as np
from notebookdata import mnist_diy
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class housing_dataset:
    def __init__(self):
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        m,n = housing.data.shape
        self.X_bais = np.c_[np.ones((m,1)),housing.data]
        self.X_bais = StandardScaler().fit_transform(self.X_bais)
        self.y = housing.target
        print("type(X_bais)=",type(self.X_bais)," shape=",self.X_bais.shape)
        # trainlen = len(self.X_bais)*0.8
        # self.X_bais_train = self.X_bais[:trainlen]
        # self.X_bais_test = self.X_bais[trainlen:]
        # self.y_train = self.y[:trainlen]
        # self.y_test = self.y[trainlen:]

        # self.X = tf.constant(self.housing_data_plus_bias_scaled,dtype=tf.float32,name="X")
        # self.y = tf.constant(housing.target.reshape(-1,1),dtype=tf.float32,name="y")




class mnist_data:
    def __init__(self):
        traindata = list(mnist_diy.read(dataset="training",path="../notebookdata"))
        testdata =list(mnist_diy.read(dataset="testing",path="../notebookdata"))
        y_train =[x[0] for x in traindata]
        X_train =[x[1] for x in traindata]
        y_test =[x[0] for x in testdata]
        X_test =[x[1] for x in testdata]
        self.X_train = np.array(X_train).reshape(len(traindata),28*28).astype(np.float32)
        self.X_train_bais =np.c_[np.ones((len(self.X_train),1)),self.X_train].astype(np.float32)
        self.y_train =np.array(y_train).reshape(len(y_train),1).astype(np.int32)
        self.X_test =np.array(X_test).reshape(len(X_test),28*28).astype(np.float32)
        self.X_test_bais =np.c_[np.ones((len(self.X_test),1)),self.X_test].astype(np.float32)
        self.y_test =np.array(y_test).reshape(len(y_test),1).astype(np.int32)
        self.X_train_bais_scaled =StandardScaler().fit_transform(self.X_train_bais)
        self.X_test_bais_scaled = StandardScaler().fit_transform(self.X_test_bais)
        self.X_train_scaled = StandardScaler().fit_transform(self.X_train)
        self.X_test_scaled =  StandardScaler().fit_transform(self.X_test)
        print("type=",self.y_train.dtype)


def test_tensor_highapi():
    # feature_columns = g_mnist_data.X_test_bais.shape[1]
    feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(g_mnist_data.X_train_bais_scaled)
    print("feature_columns=",feature_columns)
    dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[30,10],n_classes = 10,feature_columns=feature_columns)
    dnn_clf.fit(x=g_mnist_data.X_train_bais_scaled,y=g_mnist_data.y_train,batch_size=50,steps=10000)
    print("fit end")
    y_pred = list(dnn_clf.predict(g_mnist_data.X_test_bais))
    print("accuracy=",accuracy_score(g_mnist_data.y_test,y_pred))


class myfirstmode:
    def __init__(self):
        print("myfirstmode init")
        hidden1_n = 20
        hidden2_n = 10
        output_n = 10    
        self.test_var = tf.Variable([1,2,3])
        self.X = tf.placeholder(shape=(None,28*28),dtype=tf.float32)
        self.y = tf.placeholder(shape=(None),dtype=tf.int32)
        with tf.name_scope("dnn"):
            hidden1 = fully_connected(self.X,hidden1_n,scope="hidden1")
            hidden2 = fully_connected(hidden1,hidden2_n,scope="hidden2")
            logits = fully_connected(hidden2,output_n,scope="output",activation_fn=None)
        self.logits =logits

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            loss = tf.reduce_mean(xentropy,name="loss")
        self.loss = loss

    def fit_bach(self,X,y):
        print("begin fit,shape(X)={},shape(y)={}".format(X.shape,y.shape))
        learning_rate = 0.05
        n_epochs = 200
        n_bachsize = 200
        saver = tf.train.Saver()
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.training_op = optimizer.minimize(self.loss)

        test_var_op = self.test_var.assign(self.test_var+1)
        init = tf.global_variables_initializer()
        with tf.Session() as sess0:
            print("begin train  session")
            init.run()
            for epoch in range(n_epochs):
                rand_index = np.random.permutation(len(X))
                X_bachs = np.array_split(X[rand_index],n_bachsize)
                y_bachs = np.array_split(y[rand_index],n_bachsize)
                for xbach,ybach in  zip(X_bachs,y_bachs):
                    self.training_op.run(feed_dict={self.X:xbach,self.y:ybach})
                # print("{} train voer,loss={},shape(loss)={}".format(epoch,self.loss.eval(feed_dict={self.X:X,self.y:y}),self.logits.get_shape()))
                print("{} train2 voer,loss={},shape(loss)={}".format(epoch,self.loss.eval(feed_dict={self.X:xbach,self.y:ybach}),self.logits.eval(feed_dict={self.X:xbach,self.y:ybach}).shape))

            saver.save(sess0,"./mymodel.ckpt")




    def fit(self,X,y):
        print("begin fit,shape(X)={},shape(y)={}".format(X.shape,y.shape))
        learning_rate = 0.05
        n_epochs = 200
        n_bachsize = 100
        saver = tf.train.Saver()
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.training_op = optimizer.minimize(self.loss)
            
        test_var_op = self.test_var.assign(self.test_var+1)
        init = tf.global_variables_initializer()
        with tf.Session() as sess0:
            print("begin train  session")
            init.run()
            for epoch in range(n_epochs):
                self.training_op.run(feed_dict={self.X:X,self.y:y})
                sess0.run(test_var_op)
                print("{} train voer,loss={}".format(epoch,self.loss.eval(feed_dict={self.X:X,self.y:y})))
                # print("tmp_test_var=",tmp_test_var.eval())
                print("self_test_var=",self.test_var.eval())
            saver.save(sess0,"./mymodel.ckpt")

    def predict(self,X):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,"./mymodel.ckpt")
            print("predict test_var={}".format(self.test_var.eval()))
            proability = self.logits.eval(feed_dict={self.X:X})
            y_pred =np.argmax(proability,axis=1)
        return y_pred



class CnnModel():
    def __init__(self):
        print("CnnTest init")
        self.modelfile = "./CnnModel.ckpt"
        channels_n = 1
        self.filter_n = 2

        hidden1_n = 30
        hidden2_n = 10
        output_n = 10  

        filter_test= np.zeros(shape=(7,7,channels_n,self.filter_n))
        filter_test[:,3,:,0] = 1
        filter_test[3,:,:1] =1
        self.filters = filter_test
        self.X = tf.placeholder(shape=(None,28*28),dtype=tf.float32)
        self.X_c = tf.reshape(self.X, [-1,28,28,channels_n]) 
        self.y = tf.placeholder(shape=(None),dtype=tf.int32)
        print("begin construct convolution")
        self.convolution = tf.nn.conv2d(self.X_c,self.filters,strides=[1,2,2,1],padding="SAME")
        with tf.name_scope("dnn"):
            self.hidden1 = fully_connected(tf.reshape(self.convolution,[-1,14*14*self.filter_n]),hidden1_n,scope="hidden1")
            self.hidden2 = fully_connected(self.hidden1,hidden2_n,scope="hidden2")
            logits = fully_connected(self.hidden2,output_n,scope="output",activation_fn=None)
        self.logits = logits

        with tf.name_scope("loss"):
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            loss = tf.reduce_mean(xentropy,name="loss")
        self.loss = loss

    def fit_bach(self,X,y):
        print("begin fit,shape(X)={},shape(y)={}".format(X.shape,y.shape))
        learning_rate = 0.05
        n_epochs = 100
        n_bachsize = 200
        saver = tf.train.Saver()
        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            self.training_op = optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        with tf.Session() as sess0:
            print("begin train  session")
            init.run()
            for epoch in range(n_epochs):
                rand_index = np.random.permutation(len(X))
                X_bachs = np.array_split(X[rand_index],n_bachsize)
                y_bachs = np.array_split(y[rand_index],n_bachsize)
                for xbach,ybach in  zip(X_bachs,y_bachs):
                    run_op,run_loss = sess0.run([self.training_op,self.loss],feed_dict={self.X:xbach,self.y:ybach})
                print("{} train voer,loss={}".format(epoch,run_loss))
            saver.save(sess0,self.modelfile)


    def predict(self,X):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,self.modelfile)
            print("cnn predict")
            proability = self.logits.eval(feed_dict={self.X:X})
            y_pred =np.argmax(proability,axis=1)
        return y_pred
    


class RecurrentModel:
    def __init__(self):
        self.X = tf.placeholder(shape=(None,28*28),dtype=tf.float32)
        self.y = tf.placeholder(shape=(None),dtype= tf.int32)
        self.n_steps = 28
        self.n_inputs = 28
        self.n_neurons = 100
        self.n_outputs = 10
        self.learning_rate = 0.05
        self.modelfile ="./recuurent.ckpt"
        self.epochs =2
        self.n_bachsize = 100
        self.n_layers = 3

        self.X_c = tf.reshape(self.X,[-1,self.n_steps,self.n_inputs])
        # basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.n_neurons)
        # self.cells = [tf.contrib.rnn.BasicLSTMCell(self.n_neurons) for _ in range(self.n_layers)]
        self.cells = [tf.contrib.rnn.BasicRNNCell(self.n_neurons) for _ in range(self.n_layers)]
        multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.cells,state_is_tuple=False)
        self.outputs,self.states = tf.nn.dynamic_rnn(multi_layer_cell,self.X_c,dtype=tf.float32)
        self.logits = fully_connected(self.states,self.n_outputs,activation_fn=None)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits)
        self.loss = tf.reduce_mean(xentropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        correct = tf.nn.in_top_k(self.logits,self.y,1)
        self.accuracy_score = tf.reduce_mean(tf.cast(correct,tf.float32))

        
    
    def fit_bach(self,X,y):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init.run()
            for epoch in range(self.epochs):
                rand_index = np.random.permutation(len(X))
                X_bachs = np.array_split(X[rand_index],self.n_bachsize)
                y_bachs = np.array_split(y[rand_index],self.n_bachsize)
                for xbach,ybach in  zip(X_bachs,y_bachs):
                    train_op_run,loss_run = sess.run([self.train_op,self.loss],feed_dict={self.X:xbach,self.y:ybach})
                print("rnn loss=",loss_run)
            saver.save(sess,self.modelfile)

    
    def predict(self,X):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,self.modelfile)
            logits_run,state_run =sess.run([self.logits,self.states],feed_dict={self.X:X})
            y_pred =np.argmax(logits_run,axis=1)
            return y_pred

    

class AutoCoderMode:
    def __init__(self):
        n_inputs = 28*28
        n_hidden1 = 30
        n_hidden2 = 10#coding
        n_hidden3 = n_hidden1
        n_outputs = n_inputs
        self.epochs = 1000
        self.modelfile="/tmp/autoencoder.tf"
        self.n_bachsize = 50

        # initial_learning_rate = 0.1
        # decay_steps = 10000
        # decay_rate = 1/10
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,decay_steps, decay_rate)
        learning_rate = 0.05

        l2_reg =0.01

        self.X = tf.placeholder(shape=(None,n_inputs),dtype=tf.float32)
        self.X_c = tf.placeholder(shape=(None,n_inputs),dtype=tf.float32)

        with tf.contrib.framework.arg_scope(
            [fully_connected],
            activation_fn = tf.nn.elu,
            weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
            # weights_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
        ):
            hidden1 = fully_connected(self.X_c,n_hidden1)
            hidden2 = fully_connected(hidden1,n_hidden2)#coding
            hidden3 = fully_connected(hidden2,n_hidden3)
            self.outputs = fully_connected(hidden3,n_outputs,activation_fn=None)
        
        reconstruction_loss = tf.reduce_mean(tf.square(self.outputs -self.X))
        self.reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = tf.add_n([reconstruction_loss] + self.reg_loss)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(self.loss)

        if False == os.path.exists(self.modelfile+".index"):
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            with tf.Session() as sess:
                init.run()
                saver.save(sess,self.modelfile)
        else:
            print("yes find file")

    def fit_bach(self,X):
        self.global_loss = 0
        # init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        self.fit = StandardScaler().fit(X)
        X_c = self.fit.transform(X)
        with tf.Session() as sess:
            # init.run()
            saver.restore(sess,self.modelfile)
            for epoch in range(self.epochs):
                rand_index = np.random.permutation(len(X_c))
                X_bachs = np.array_split(X[rand_index],self.n_bachsize)
                X_c_bachs = np.array_split(X_c[rand_index],self.n_bachsize)
                for xbach,x_c_bachs in  zip(X_bachs,X_c_bachs):
                    train_op_run,loss_run,reg_loss_run = sess.run([self.train_op,self.loss,self.reg_loss],feed_dict={self.X:xbach,self.X_c:x_c_bachs})
                    print("autorecoder,epoch =",epoch,"loss=",loss_run," reg_loss=",reg_loss_run)
                g_loss = sess.run(self.loss,feed_dict={self.X:X,self.X_c:X_c})
                if(0 == self.global_loss  or g_loss < self.global_loss ):
                    self.global_loss = g_loss
                    saver.save(sess,self.modelfile)
                    print("save loss=",self.global_loss)
                
            


    def predict(self,X):
        # X_c = self.fit.transform(X)
        X_c = StandardScaler().fit_transform(X)
        # init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            print("begin sess")
            saver.restore(sess,self.modelfile)
            # init.run()
            outputs_val = self.outputs.eval(feed_dict={self.X_c:X_c})
        print("begin recurrent plot")
        for i in range(len(X)):
            print("begin plot,i=",i)
            plt.subplot(len(X),2,i*2+1)
            plot_image(X[i])
            plt.subplot(len(X),2,i*2+2)
            plot_image(outputs_val[i])
        plt.show()


    




class CnnTest():
    def __init__(self):
        print("CnnTest init")
        channels_n = 1
        filter_n = 2

        filter_test= np.zeros(shape=(7,7,channels_n,filter_n))
        filter_test[:,3,:,0] = 1
        filter_test[3,:,:1] =1
        self.filters = filter_test
        self.X = tf.placeholder(shape=(None,28,28,channels_n),dtype =tf.float32)
        self.convolution = tf.nn.conv2d(self.X,self.filters,strides=[1,2,2,1],padding="SAME")

    def fit(self,X):
        with tf.Session() as sess:
            output = sess.run(self.convolution,feed_dict={self.X:X})
        plt.imshow(output[0,:,:,1],cmap=plt.cm.gray)
        plt.show()


def CnnTest_fun():
    myimage = g_mnist_data.X_train[10].reshape(1,28,28,1)
    plt.imshow(myimage.reshape(28,28),plt.cm.gray)
    plt.show()
    print("type(myimage)=",myimage.shape)
    cnn = CnnTest()
    cnn.fit(myimage)


class PoolTest():
    def __init__(self):
        print("Pool init")
        channels_n = 1
        
        self.X = tf.placeholder(shape=(None,28,28,channels_n),dtype =tf.float32)
        self.pool = tf.nn.max_pool(self.X,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

    def fit(self,X):
        with tf.Session() as sess:
            output = sess.run(self.pool,feed_dict={self.X:X})
        print("shape(output)=",output.shape)
        plt.imshow(output[0,:,:,0],cmap=plt.cm.gray)
        plt.show()

def PoolTest_fun():
    myimage = g_mnist_data.X_train[10].reshape(1,28,28,1)
    pooltest = PoolTest()
    pooltest.fit(myimage)


        

def plot_image(image,shape=[28,28]):
    plt.imshow(image.reshape(shape),cmap="Greys",interpolation="nearest")
    plt.axis("off")





def mymodetest():
    print("type(X=)={},type(y)={}".format(g_mnist_data.X_train.shape,g_mnist_data.y_train.shape))
    dnn_clf = myfirstmode()
    print("X={},y={}".format(g_mnist_data.X_train_scaled,g_mnist_data.y_train.reshape(-1,)))
    dnn_clf.fit_bach(X=g_mnist_data.X_train_scaled,y=g_mnist_data.y_train.reshape(-1,))
    print("fit end")
    
    standerfit_model = StandardScaler().fit(g_mnist_data.X_train)
    diy_X_test_scaled = standerfit_model.transform(g_mnist_data.X_test)
    y_pred = list(dnn_clf.predict(diy_X_test_scaled))
    print("type(y_pred)={}".format(type(y_pred)))

    print("accuracy=",accuracy_score(g_mnist_data.y_test.reshape((-1,)),y_pred))

def directpredict():
    print("directpredict")
    standerfit_model = StandardScaler().fit(g_mnist_data.X_train)
    diy_X_test_scaled = standerfit_model.transform(g_mnist_data.X_test)
    new_dnn_clf = myfirstmode()
    y_pred = list(new_dnn_clf.predict(diy_X_test_scaled))
    print("directpredict accuracy=",accuracy_score(g_mnist_data.y_test.reshape((-1,)),y_pred))



def CnnModelTest():
    print("CnnModelTest enter")
    dnn_clf = CnnModel()
    print("X={},y={}".format(g_mnist_data.X_train_scaled,g_mnist_data.y_train.reshape(-1,)))
    dnn_clf.fit_bach(X=g_mnist_data.X_train_scaled,y=g_mnist_data.y_train.reshape(-1,))
    print("cnn fit end")
    
    standerfit_model = StandardScaler().fit(g_mnist_data.X_train)
    diy_X_test_scaled = standerfit_model.transform(g_mnist_data.X_test)
    y_pred = list(dnn_clf.predict(diy_X_test_scaled))
    print("cnn accuracy=",accuracy_score(g_mnist_data.y_test.reshape((-1,)),y_pred))


def RecurrentModelTest():
    print("RecurrentModelTest enter")
    recurent_clf = RecurrentModel()
    print("X={},y={}".format(g_mnist_data.X_train_scaled,g_mnist_data.y_train.reshape(-1,)))
    recurent_clf.fit_bach(X=g_mnist_data.X_train_scaled,y=g_mnist_data.y_train.reshape(-1,))
    print("recurrent fit end")
    
    standerfit_model = StandardScaler().fit(g_mnist_data.X_train)
    diy_X_test_scaled = standerfit_model.transform(g_mnist_data.X_test)
    y_pred = list(recurent_clf.predict(diy_X_test_scaled))
    # print("y_pred=",y_pred)
    print("recureent accuracy=",accuracy_score(g_mnist_data.y_test.reshape((-1,)),y_pred))


def MutRnnTmpTest():
    print("MutRnnTmpTest")
    n_bathsize = 3
    n_steps = 5
    n_fetures = 6
    n_layer = 2
    n_nurons = 10
    X = tf.random_normal(shape=[n_bathsize,n_steps,n_fetures], dtype=tf.float32)
    X = tf.reshape(X, [-1, n_steps, n_fetures])
    # cell = tf.nn.rnn_cell.BasicLSTMCell(10)
    cells = [tf.contrib.rnn.BasicLSTMCell(n_nurons) for _ in range(n_layer)]
    lstm_multi = tf.nn.rnn_cell.MultiRNNCell(cells,  state_is_tuple=True)
    state = lstm_multi.zero_state(n_bathsize, tf.float32)
    output, state = tf.nn.dynamic_rnn(lstm_multi, X, initial_state=state, time_major=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("output'shape=",output.get_shape()," state",state)
        # print(sess.run(state))


def autoencoderTest():
    print("autoencoderTest enter")
    autoencoder_clf = AutoCoderMode()
    print("X={},y={}".format(g_mnist_data.X_train_scaled,g_mnist_data.y_train.reshape(-1,)))
    autoencoder_clf.fit_bach(X=g_mnist_data.X_train)
    print("autoencoder fit end")
    autoencoder_clf.predict(g_mnist_data.X_test[:3])


    
# g_housedata = housing_dataset()
g_mnist_data = mnist_data()
if __name__== "__main__":
    print("hello world")
    # test_tensor_highapi()
    mymodetest()
    # directpredict()
    # CnnTest_fun()
    # PoolTest_fun()
    # CnnModelTest()
    # RecurrentModelTest()
    # MutRnnTmpTest()
    # autoencoderTest()
