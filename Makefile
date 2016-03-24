CC=g++
CFLAGS += -O3 -g
CXXFLAGS += -std=c++11 -O3 -fopenmp -g -I/usr/local/cuda/include #-DCPU_ONLY=1 #-I/opt/caffe-fcn/include -I/usr/local/cuda/include
LDFLAGS += -fopenmp -L/usr/lib64
#LDLIBS +=  -lxgboost /usr/local/lib/dmlc_simple.o -lrabit -Wl,--whole-archive -lcaffe -Wl,--no-whole-archive -lproto -lprotobuf -lsnappy -lgflags -lglog -lleveldb -llmdb -lunwind -lhdf5_hl -lhdf5 -lopencv_features2d -lopencv_imgproc -lopencv_imgcodecs -lopencv_flann -lopencv_core -lopencv_hal -lIlmImf -lippicv -lboost_timer -lboost_chrono -lboost_program_options -lboost_log -lboost_log_setup -lboost_thread -lboost_filesystem -lboost_system -lopenblas -ljpeg -ltiff -lpng -ljasper -lwebp -lpthread -lz -lm -lrt -ldl
LDLIBS =  -lcaffe $(shell pkg-config --libs opencv) \
	 -lboost_timer -lboost_chrono -lboost_thread -lboost_filesystem -lboost_system -lboost_program_options  \
	 -lglog

COMMON = xnn.o caffe.o mxnet.o
PROGS = visualize predict #caffex-extract	caffex-predict batch-resize import-images

all:	$(PROGS)

predict:	predict.cpp $(COMMON)

visualize:	visualize.cpp $(COMMON)

