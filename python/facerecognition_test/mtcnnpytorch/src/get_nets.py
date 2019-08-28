import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import logging


modelbasepath="/home/sailingzhang/winshare/develop/source/mygit/test/python/facerecognition_test/mtcnnpytorch/"

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)




def create_pytorch_mtcnn():
    pnet = MyPNet_tensorflowdata()
    rnet = MyRNet_tensorflowdata()
    onet = MyONet_tensorflowdata()
    onet.eval()
    return pnet,rnet,onet





class MyPNet_tensorflowdata(nn.Module):
    
    def __init__(self,channel=3):

        super(MyPNet_tensorflowdata, self).__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W
# https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
# o = output
# p = padding
# k = kernel_size
# s = stride
# d = dilation
# o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
#计算 padding="same"  的公式        
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(channel, 10, 3, 1)),
            ('PReLU1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2,padding=1)),#same padding

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('PReLU2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('PReLU3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1,padding=1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1,padding=1)

        weights = np.load(modelbasepath+'src/weights/pnet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        logging.debug("mypnet src x.shape={}".format(x.shape))
        x = x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        # x = x.reshape((-1,3,x.shape[2],x.shape[1])).astype(np.float32)

        logging.debug("mypnet after x.shape={},x.dtype={}".format(x.shape,x.dtype))
        x = torch.from_numpy(x)
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)

        b = b.detach().numpy()
        b= b.reshape((-1,b.shape[2],b.shape[3],4))
        a = a.detach().numpy()
        a= a.reshape((-1,a.shape[2],a.shape[3],2))
        logging.debug("mypnet,b.shape={},a.shape={}".format(b.shape,a.shape))
        return b, a


class MyRNet_tensorflowdata(nn.Module):

    def __init__(self):

        super(MyRNet_tensorflowdata, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2)),

            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2)),

            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(modelbasepath+'src/weights/rnet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        logging.debug("myrnet src x.shape={}".format(x.shape))
        # x =x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        x =x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        logging.debug("myrnet after x.shape={}".format(x.shape))
        x = torch.from_numpy(x)
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)

        
        b = b.detach().numpy()
        # b= b.reshape((-1,4))
        a = a.detach().numpy()
        # a= a.reshape((-1,2))

        logging.debug("myrnet,b.shape={},a.shape={}".format(b.shape,a.shape))
        return b, a


class MyONet_tensorflowdata(nn.Module):

    def __init__(self):

        super(MyONet_tensorflowdata, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(modelbasepath+'src/weights/onet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        logging.debug("myonet src x.shape={}".format(x.shape))
        # x=x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        x=x.reshape((-1,3,x.shape[2],x.shape[1])).astype(np.float32)
        logging.debug("myonet after x.shape={}".format(x.shape))
        x = torch.from_numpy(x)
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)

        c = c.detach().numpy()
        b = b.detach().numpy()
        a = a.detach().numpy()

        logging.debug("myonet,c.shape={},b.shape={},a.shape={}".format(c.shape,b.shape,a.shape))
        return c, b, a




















class MyPNet_1(nn.Module):
    
    def __init__(self):

        super(MyPNet, self).__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load(modelbasepath+'src/weights/pnet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        logging.debug("mypnet src x.shape={}".format(x.shape))
        x = x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        # x = x.reshape((-1,3,x.shape[2],x.shape[1])).astype(np.float32)

        logging.debug("mypnet after x.shape={},x.dtype={}".format(x.shape,x.dtype))
        x = torch.from_numpy(x)
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)

        b = b.detach().numpy()
        b= b.reshape((-1,b.shape[2],b.shape[3],4))
        a = a.detach().numpy()
        a= a.reshape((-1,a.shape[2],a.shape[3],2))
        logging.debug("mypnet,b.shape={},a.shape={}".format(b.shape,a.shape))
        return b, a


class MyRNet_1(nn.Module):

    def __init__(self):

        super(MyRNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(modelbasepath+'src/weights/rnet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        logging.debug("myrnet src x.shape={}".format(x.shape))
        # x =x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        x =x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        logging.debug("myrnet after x.shape={}".format(x.shape))
        x = torch.from_numpy(x)
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)

        
        b = b.detach().numpy()
        # b= b.reshape((-1,4))
        a = a.detach().numpy()
        # a= a.reshape((-1,2))

        logging.debug("myrnet,b.shape={},a.shape={}".format(b.shape,a.shape))
        return b, a


class MyONet_1(nn.Module):

    def __init__(self):

        super(MyONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(modelbasepath+'src/weights/onet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        logging.debug("myonet src x.shape={}".format(x.shape))
        # x=x.reshape((-1,3,x.shape[1],x.shape[2])).astype(np.float32)
        x=x.reshape((-1,3,x.shape[2],x.shape[1])).astype(np.float32)
        logging.debug("myonet after x.shape={}".format(x.shape))
        x = torch.from_numpy(x)
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)

        c = c.detach().numpy()
        b = b.detach().numpy()
        a = a.detach().numpy()

        logging.debug("myonet,c.shape={},b.shape={},a.shape={}".format(c.shape,b.shape,a.shape))
        return c, b, a


class PNet(nn.Module):

    def __init__(self):

        super(PNet, self).__init__()

        # suppose we have input with size HxW, then
        # after first layer: H - 2,
        # after pool: ceil((H - 2)/2),
        # after second conv: ceil((H - 2)/2) - 2,
        # after last conv: ceil((H - 2)/2) - 4,
        # and the same for W

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

        weights = np.load(modelbasepath+'src/weights/pnet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)
        return b, a


class RNet(nn.Module):

    def __init__(self):

        super(RNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

        weights = np.load(modelbasepath+'src/weights/rnet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)
        return b, a


class ONet(nn.Module):

    def __init__(self):

        super(ONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

        weights = np.load(modelbasepath+'src/weights/onet.npy')[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        # a = F.softmax(a)
        a = F.softmax(a,dim=1)
        return c, b, a
