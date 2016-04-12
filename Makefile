.PHONY:	all install
CC=gcc
CXX=g++
CFLAGS += -O3 -g
CXXFLAGS += -DUSE_CAFFE=1 -std=c++11 -O3 -fopenmp -g -I/usr/include/python2.7 -I/usr/local/cuda/include #-DCPU_ONLY=1 #-I/opt/caffe-fcn/include -I/usr/local/cuda/include
LDFLAGS += -fopenmp -L/usr/lib64 
#LDLIBS = libxnn.a -lcaffe \
	 $(shell pkg-config --libs opencv) \
	 -ljson11 \
	 -lboost_timer -lboost_chrono -lboost_thread -lboost_filesystem -lboost_system -lboost_program_options  \
	 -lglog #-lpython2.7

LDLIBS = libxnn.a -lcaffe $(shell pkg-config --libs opencv) \
	 -ljson11 -lboost_timer -lboost_chrono -lboost_thread -lboost_filesystem -lboost_system -lboost_program_options -lglog 

COMMON = libxnn.a
PROGS = predict #xnn-roc #test_python # visualize predict #caffex-extract	caffex-predict batch-resize import-images

all:	$(PROGS)

libxnn.a:	xnn.o caffe.o # mxnet.o python.o
	ar rvs $@ $^

PROGS:	%:	%.o $(COMMON)

install:	libxnn.a
	cp libxnn.a /usr/local/lib
	cp xnn.h /usr/local/include
