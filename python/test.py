import os
import sys
import logging
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


if __name__ == "__main__":
    log_init("test.log")
    # filterTest()
    yieldtest()