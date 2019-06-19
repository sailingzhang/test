import os
import sys
import logging
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append("..")
from log_init import log_init

def filterTest():
    onlist=[x for x in range(10)]
    newlist =list(filter(lambda x:(x%2==0),onlist))
    logging.debug("newlist={}".format(newlist))
    newlist = list(map(lambda x:x*2,onlist))
    logging.debug("newlist={}".format(newlist))

def myyield(n):
    i =n
    while i>0:
        yield i
        i-=1

def yieldtest():
    f = myyield(2)
    for i in f:
        logging.debug("i={}".format(i))
    for j in myyield(3):
        logging.debug("j={}".format(j))


def gatherTest():
    t1 = torch.Tensor([[1,2,3],[4,5,6]])
    logging.debug("t1.size={}".format(t1.size()))
    logging.debug("t1={}".format(t1))
    newt1 = torch.gather(t1,dim=1,index=torch.LongTensor([[0,1],[2,0]]))
    logging.debug("newt1={}".format(newt1))

    newt2 =torch.gather(t1,dim=0,index=torch.LongTensor([[0,1,1],[0,0,0]]))
    logging.debug("newt2={}".format(newt2))


def squeezeTest():
    sq1 =torch.Tensor([[[1,2,3]],[[4,5,6]]])
    logging.debug("sq1'size={}".format(sq1.size()))
    asq2 = sq1.squeeze(1)
    logging.debug("asq2={}".format(asq2))
    asq3 = asq2.unsqueeze(2)
    logging.debug("asq3={}".format(asq3))
    
def maxTest():
    one = torch.Tensor([[2,3,1,5],[3,3,5,1]])
    one_0 =one.max(0)
    one_1 = one.max(1)
    logging.debug("one_0={},one_1={}".format(one_0,one_1))


	

if __name__ == "__main__":
    log_init("test.log")
    # filterTest()
    # yieldtest()
    # gatherTest()
    # squeezeTest()
    maxTest()