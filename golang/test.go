package main

import(
	"github.com/cihub/seelog"
	"path/filepath"
	"path"
	"os"
	"sync"
	"time"
	"strconv"
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


func main(){
	LOG_INIT("test_log_conf.xml")
	defer seelog.Flush()
	// rwtest()
	mapsliceTest2()
}

