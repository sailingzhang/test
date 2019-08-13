import os
import sys
import pickle
import logging
import numpy as np
import shutil
from scipy import misc

class trainDataBase:
    def __init__(self):
        pass
    def AddData(self,yId,xId,srcx,embx):
        '''
        srcx:bytes[]
        embx:np.arrarys
        '''
        return None
    def RemoveData(self,yId,xId):
        return None
    def RemoveClass(self,yId):
        return None
    def clear(self):
        return
    def GetBatchData(self):
        srcX = None
        embX = None
        y = None        
        xIds = None
        yIds= None
        return (srcX,embX,y,xIds,yIds)
    def GetMiniBachData(self,batchSize,feednum):
        srcX = None
        embX = None
        xIds = None
        y = None
        yIds= None
        return (srcX,embX,xIds,y,yIds)
    def show(self):
        return



class lableClass:
    def __init__(self,yId):
        self.yId = yId
        self.items={}
    def add(self,xId,srcx,embx):
        item = self.items.get(xId,[None,None])
        if srcx is not None:
            item[0] = srcx
        if embx is not None:
            item[1] = embx
        self.items[xId]= item
        
class LocalDataManger(trainDataBase):
    def __init__(self,savePathDir,srcSuffix=".src",embSuffix=".emb",isSyncSrcx=False,int_dimensionality = 512):
        logging.debug("enter,savePathDir={},len(savePathDir)={}".format(savePathDir,len(savePathDir)))
        self.data ={}
        self.savePathDir = os.path.expanduser(savePathDir)
        self.srcSuffix = srcSuffix
        self.embSuffix = embSuffix
        self.isSyncSrcx = isSyncSrcx
        self.maxbakDirNum = 3
        self.bStaySrcInMemory = False
        if "" == self.savePathDir:
            logging.info("exit,no savepath")
            return
        if not os.path.exists(self.savePathDir):
            os.makedirs(self.savePathDir)
            logging.debug("exit...no exit savepath={},so create".format(self.savePathDir))
            return
        self.clear(int_dimensionality,isRemove=False)
        # self._loaddata_()

    def _loaddata_(self):
        if "" == self.savePathDir:
            logging.debug("exit,savepath is empt,no need load")
            return
        classes = [path for path in os.listdir(self.savePathDir) if os.path.isdir(os.path.join(self.savePathDir, path))]
        logging.debug("classes={}".format(classes))
        for yId in classes:
            oneclassPath= os.path.join(self.savePathDir, yId)
            files = os.listdir(oneclassPath)
            for file in files:
                filepath = os.path.join(oneclassPath,file)
                logging.debug("filepath={}".format(filepath))
                if not os.path.isfile(filepath):
                    logging.info("the filepath={} is not file".format(filepath))
                    continue
                oneEmb = None
                oneSrc = None
                xId = None
                if filepath.endswith(self.embSuffix):
                    xId=file[0:file.find(self.embSuffix)]
                    logging.debug("load embfile={},xId={}".format(filepath,xId))
                    with open(filepath,"rb+") as f:
                        oneEmb =pickle.load(f)
                if filepath.endswith(self.srcSuffix):
                    if False == self.isSyncSrcx:
                        logging.debug("no need load file={}".format(filepath))
                        continue
                    xId=file[0:file.find(self.srcSuffix)]
                    logging.debug("load srcfile={},xId={}".format(filepath,xId))
                    with open(filepath,"rb+") as f:
                        oneSrc = f.read()

                
                logging.debug("xId={},file={}".format(xId,file))
                classInstance = self.data.get(yId,lableClass(yId))
                if self.bStaySrcInMemory:
                    classInstance.add(xId,oneSrc,oneEmb)
                else:
                    classInstance.add(xId,None,oneEmb)

                self.data[yId]=classInstance        
    def AddData(self,yId,xId,srcx,embx):
        '''
        srcx type:bytes[]
        embs np.arrays
        '''
        logging.debug("enter,yId={},xId={}".format(xId,yId))
        classInstance = self.data.get(yId,lableClass(yId))
        if self.bStaySrcInMemory:
            classInstance.add(xId,srcx,embx)
        else:
            classInstance.add(xId,None,embx)

        self.data[yId]=classInstance
        
        if "" == self.savePathDir:
            logging.info("the savepathdir is empty,so no need save")
            return
        yIdPathDir= self.savePathDir+"/"+yId
        if not os.path.exists(yIdPathDir):
            os.makedirs(yIdPathDir)
        srcxPathFile = yIdPathDir+"/"+xId+self.srcSuffix
        embxPathFile = yIdPathDir+"/"+xId+self.embSuffix
        if srcx is not None and True == self.isSyncSrcx:
            misc.imsave(srcxPathFile, srcx)
            # with open(srcxPathFile, 'wb') as f:
            #      f.write(srcx)
        if embx is not None:
            with open(embxPathFile, 'wb') as f:
                 pickle.dump(embx,f)
            
        logging.debug("exit")
        return None
    def RemoveData(self,yId,xId):
        logging.debug("enter")
        classInstance = self.data.get(yId,None)
        if classInstance is not None:
            logging.debug("begin del yId={},xId={}".format(yId,xId))
            self.data[classInstance.yId].items.pop(xId,None)
        if "" == self.savePathDir:
            logging.debug("savePath is empty,no need del file")
            return
        yIdPathDir= self.savePathDir+"/"+yId
        if not os.path.exists(yIdPathDir):
            logging.error("the dir={} is not exit".format(yIdPathDir))
            return
        srcxPathFile = yIdPathDir+"/"+xId+self.srcSuffix
        embxPathFile = yIdPathDir+"/"+xId+self.embSuffix
        if os.path.isfile(srcxPathFile):
            os.remove(srcxPathFile)
        if os.path.isfile(embxPathFile):
            os.remove(embxPathFile)
        logging.debug("exit")
        return None
    def RemoveClass(self,yId):
        logging.debug("enter")
        self.data.pop(yId,None)
        if "" == self.savePathDir:
            logging.debug("exit,savepath is empty,no need rmdir")
            return
        yIdPathDir= self.savePathDir+"/"+yId
        if os.path.exists(yIdPathDir):
            shutil.rmtree(yIdPathDir)      
        logging.debug("exit")
    def clear(self,int_dimensionality,isRemove=True):
        logging.info("savePathDir={}".format(self.savePathDir))
        self.data={}
        if True == isRemove:
            # os.rename(self.savePathDir,self.savePathDir+"_old")
            cmd ="rm -rf "+self.savePathDir+"."+str(self.maxbakDirNum)
            logging.info("cmd={}".format(cmd))
            os.system(cmd)
            for i in range(self.maxbakDirNum):
                olddir = self.savePathDir+"."+str(self.maxbakDirNum-i-1)
                newdir = self.savePathDir+"."+str(self.maxbakDirNum -i)
                cmd ="mv " + olddir+" "+newdir
                logging.info("cmd={}".format(cmd))
                os.system(cmd)
            cmd ="mv "+self.savePathDir+" "+self.savePathDir+".0"
            logging.info("cmd={},begin creat dir={}".format(cmd,self.savePathDir))
            os.system(cmd)
            os.mkdir(self.savePathDir)

        defaultdir0=self.savePathDir+"/default0"
        defaultdir1= self.savePathDir+"/default1"
        defaultfile0=defaultdir0+"/default0.emb"
        defaultfile1=defaultdir1+"/default1.emb"

        if not os.path.exists(self.savePathDir+"/default0"):
            os.mkdir(self.savePathDir+"/default0")
        if not os.path.exists(self.savePathDir+"/default1"):
            os.mkdir(self.savePathDir+"/default1")
        with open(defaultfile0, 'wb') as f:
            default0emb = np.zeros(shape=(1,int_dimensionality))
            pickle.dump(default0emb,f)
        with open(defaultfile1, 'wb') as f:
            default1emb = np.ones(shape=(1,int_dimensionality))
            pickle.dump(default1emb,f)
        self._loaddata_()
        return
    def GetClass(self,yId):
        getClass=self.data.get(yId,None)
        return getClass
    def GetBatchData(self):
        logging.debug("enter")
        srcX = []
        embX = None
        y = None
        xIds = []
        yIds= []
        index = 0
        allClasses = self.data.values()
        logging.debug("type(allClasses)={}".format(type(allClasses)))
        for oneClassInstance in allClasses:
            get_yId = oneClassInstance.yId
            yIds.append(get_yId)
            indexArr = np.array([index])
            index+=1
            for (instanceKey,instanceValue) in oneClassInstance.items.items():
                get_xId = instanceKey
                get_srcX= instanceValue[0]
                get_embx = instanceValue[1]
                get_embx = get_embx.reshape(1,-1)
                logging.debug("shape(get_embx)={},get_xId={},get_yId={}".format(get_embx.shape,get_xId,get_yId))
                if embX is None:
                    embX = get_embx
                else:
                    embX = np.append(embX,get_embx,axis=0)
                if y is None:
                    y = np.array(indexArr)
                else:
                    y= np.append(y,indexArr,axis=0)
                
                srcX.append(get_srcX)
                xIds.append(get_xId)
        logging.debug("y={}".format(y))
        logging.debug("exit,shape(embX)={},shape(y)={},len(srcX)={},len(xIds)={},len(yIds)={}".format(embX.shape,y.shape,len(srcX),len(xIds),len(yIds)))
        return (embX,y,srcX,xIds,yIds)

    def show(self):
        logging.debug("/*********************begin show***************************/")
        for (classKey,classValue) in self.data.items():
            logging.debug("yId={}".format(classValue.yId))
            for (itemKey,itemValue) in classValue.items.items():
                srcmd5="empty"
                embmd5="empty"
                srcitem = itemValue[0]
                embitem = itemValue[1]
                isSrc = srcitem is not None
                isEmb = embitem is not None
                logging.debug("xId={},xSrc={},xEmb={}".format(itemKey,isSrc,isEmb))
        logging.debug("/*********************end  show***************************/")
    
        