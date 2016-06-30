.PHONY:	all install
CC=g++
CXX=g++
CFLAGS += -O3 -g
CXXFLAGS += -DUSE_CAFFE=1 -std=c++11 -O3 -fopenmp -g -I/usr/include/python2.7 -I/usr/local/cuda/include #-DCPU_ONLY=1
#CXXFLAGS += -DUSE_PYTHON=1
LDFLAGS += -fopenmp -L/usr/lib64 
# add -lmxnet for mxnet
# add -lpython2.7 for python
LDLIBS = libxnn.a -lcaffe-picpac -lpicpac $(shell pkg-config --libs opencv) \
	 -ljson11 -lboost_timer -lboost_chrono -lboost_thread -lboost_filesystem -lboost_system -lboost_program_options -lprotoc -lprotobuf -lglog #-lpython2.7

COMMON = libxnn.a
PROGS = picpac-stream-lmdb caffe-mean test predict xnn-roc #test_python # visualize predict #caffex-extract	caffex-predict batch-resize import-images

all:	$(COMMON) $(PROGS)

libxnn.a:	xnn.o caffe.o # python.o # mxnet.o python.o
	ar rvs $@ $^

$(PROGS):	%:	%.o $(COMMON)

clean:
	rm $(PROGS) *.o

install:	libxnn.a
	cp libxnn.a /usr/local/lib
	cp xnn.h /usr/local/include
