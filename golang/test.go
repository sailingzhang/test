package main

import(
	"github.com/cihub/seelog"
	"path/filepath"
	"path"
	"os"
	"sync"
	"time"
	"strconv"
	"github.com/jdeng/goface"
)


func LOG_INIT(logconfigxml string) {
	var logConfigFile = ""
	dir, fileErr := filepath.Abs(filepath.Dir(os.Args[0]))
	if fileErr != nil {
		seelog.Errorf("err=%v", fileErr)
		logConfigFile = logconfigxml
	} else {
		logConfigFile = path.Join(dir, logconfigxml)
	}
	seelog.Tracef("file=%s", logConfigFile)
	logger, err := seelog.LoggerFromConfigAsFile(logConfigFile)
	if nil != err {
		seelog.Critical("err parsing config log file", err)
		return
	}
	seelog.ReplaceLogger(logger)
}


type rwObjectTest struct{
	intvalue int
	strvalue string
	rwmux   sync.RWMutex
}

func (this * rwObjectTest)mulRead(id int32){
	this.rwmux.RLock()
	defer this.rwmux.RUnlock()
	seelog.Tracef("readid=%d,intvalue=%d,strvalue=%s",id,this.intvalue,this.strvalue)
	<-time.After(10*time.Second)
}

func (this * rwObjectTest)mulWrite(id int32){
	this.rwmux.Lock()
	defer this.rwmux.Unlock()
	this.intvalue++
	this.strvalue = "strvalue="+strconv.Itoa(this.intvalue)
	seelog.Tracef("writeid=%d,intvalue=%d,strvalue=%s",id,this.intvalue,this.strvalue)
}


func rwtest(){
	var instance rwObjectTest

	readfun :=func(i int32){
		<-time.After(1*time.Second)
		instance.mulRead(i)
	}

	writefun := func(i int32){
		instance.mulWrite(i)
	}
	
	for i:=0;i<10;i++{
			go readfun(int32(i))			
	}

	<-time.After(1*time.Second)
	for i:=0;i<1;i++{
		go writefun(int32(i))
	}
}

func mapsliceTest(){
	oneMapSliece := make(map[int][]string)
	for i:=0;i<10;i++{
		value,have :=oneMapSliece[1]
		if false == have{
			value = make([]string,0)
		}
		value = append(value,"test")
		oneMapSliece[1]=value

	}

	seelog.Tracef("oneMapSlice=%v,len(oneMapSlice)",oneMapSliece,len(oneMapSliece[1]))
}

func mapsliceTest2(){
	oneMapSliece := make(map[int][]string)
	for i:=0;i<10;i++{
		_,have :=oneMapSliece[1]
		if false == have{
			oneMapSliece[1] = make([]string,0)
		}
		oneMapSliece[1] = append(oneMapSliece[1],"test")

	}

	seelog.Tracef("oneMapSlice=%v,len(oneMapSlice)",oneMapSliece,len(oneMapSliece[1]))
}


func faceTest(){
		// detection
		bs, err := ioutil.ReadFile(*imgFile)
		if nil != err{
			seelog.Errorf("read file err:%v",err)
			return
		}
		img, err := goface.TensorFromJpeg(bs)
		if nil != err{
			seelog.Errorf("tensor form jpg err:%v",err)
			return
		}
		det, err := goface.NewMtcnnDetector("mtcnn.pb")
		if nil != err{
			seelog.Errorf("new detector err:%v",err)
			return
		}
		bbox, err := det.DetectFaces(img) //[][]float32, i.e., [x1,y1,x2,y2],...
		if nil != err{
			seelog.Errorf("detect err:",err)
			return
		}
		// embeddings
		mean, std := goface.MeanStd(img)
		wimg, err := goface.PrewhitenImage(img, mean, std)
		if nil != err{
			seelog.Errorf("PrewhitenImage err:",err)
			return
		}
		fn, err := goface.NewFacenet("facenet.pb")
		if nil != err{
			seelog.Errorf("NewFacenet err:",err)
			return
		}
		emb, err := fn.Embedding(wimg)
		if nil != err{
			seelog.Errorf("Embedding err:",err)
			return
		}
}

func main(){
	LOG_INIT("test_log_conf.xml")
	defer seelog.Flush()
	// rwtest()
	mapsliceTest2()
}
