#include <vector>
#include <mvnc.h>
#include <glog/logging.h>
#define USE_MOVIDIUS 1
#include "xnn.h"
#include "fp16.h"

namespace xnn {

#define NAME_SIZE 100

using std::vector;
typedef unsigned short half;

class MovidiusModel: public Model {
    void *LoadFile(const char *path, unsigned int *length)
    {
        FILE *fp;
        char *buf;

        fp = fopen(path, "rb");
        if(fp == NULL)
            return 0;
        fseek(fp, 0, SEEK_END);
        *length = ftell(fp);
        rewind(fp);
        if(!(buf = (char*) malloc(*length)))
        {
            fclose(fp);
            return 0;
        }
        if(fread(buf, 1, *length, fp) != *length)
        {
            fclose(fp);
            free(buf);
            return 0;
        }
        fclose(fp);
        return buf;
    }
protected:
    void *deviceHandle;
    void* graphHandle;
    vector<float> imagebuf;
    vector<half> halfbuf;
    static const int networkDim = 224;
public:
    MovidiusModel (fs::path const& dir)
    {
        shape[0] = 1;
        shape[1] = 3;
        shape[2] = networkDim;
        shape[3] = networkDim;
        mvncStatus retCode;
        char devName[NAME_SIZE];
        retCode = mvncGetDeviceName(0, devName, NAME_SIZE);
        CHECK(retCode == MVNC_OK);
        
        // Try to open the NCS device via the device name
        retCode = mvncOpenDevice(devName, &deviceHandle);
        CHECK(retCode == MVNC_OK);

        // Now read in a graph file
        unsigned int graphFileLen;
        void* graphFileBuf = LoadFile((dir/"graph").native().c_str(), &graphFileLen);

        // allocate the graph
        retCode = mvncAllocateGraph(deviceHandle, &graphHandle, graphFileBuf, graphFileLen);
        CHECK(retCode == MVNC_OK);
        free(graphFileBuf);

        means[0] = means[1] = means[2] = 0;

        fs::path mean_file = dir / "mean";
        fs::ifstream test(mean_file);
        if (test) {
            test >> means[0];
            means[1] = means[2] = means[0];
            test >> means[1];
            test >> means[2];
        }

        imagebuf.resize(networkDim * networkDim * 3);
        halfbuf.resize(imagebuf.size());
    }

    ~MovidiusModel () {
        mvncDeallocateGraph(graphHandle);
        mvncCloseDevice(deviceHandle);
    }

    virtual void apply (vector<cv::Mat> const &images, vector<float> *ft) {
        mvncStatus retCode;
        int batch = shape[0];
        CHECK(!images.empty()) << "must input >= 1 images";
        int off = 0;
        for (int i = 0; i < images.size(); ++i) {
            CHECK(0) << "Need to update preprocess to work with NHWC";
            float *e = preprocess(images[i], &imagebuf[0]);
	        floattofp16((unsigned char *)&halfbuf[0], &imagebuf[0], imagebuf.size());
            retCode = mvncLoadTensor(graphHandle, &halfbuf[0], sizeof(halfbuf[0])*halfbuf.size(), NULL);
            CHECK(retCode == MVNC_OK);
            void* resultData16;
            void* userParam;
            unsigned int lenResultData;
            retCode = mvncGetResult(graphHandle, &resultData16, &lenResultData, &userParam);
            CHECK(retCode == MVNC_OK);
            int numResults = lenResultData / sizeof(half);
            if (i == 0) {
                ft->resize(images.size() * numResults);
            }
            fp16tofloat(&ft->at(off), (unsigned char*)resultData16, numResults);
            off += numResults;
        } 
    }
};

Model *Model::create_movidius (fs::path const &dir, int batch) {
    return new MovidiusModel(dir);
}

}


