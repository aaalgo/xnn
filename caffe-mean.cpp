#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "xnn.h"

using namespace std;
using namespace caffe;

int main (int argc, char *argv[]) {
    if (argc < 2) return 0;
    string mean_file(argv[1]);

    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> meanblob;
    meanblob.FromProto(blob_proto);
    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    vector<cv::Mat> channels;
    float* data = meanblob.mutable_cpu_data();
    for (int i = 0; i < meanblob.channels(); ++i) {
        /* Extract an individual channel. */
        cv::Mat channel(meanblob.height(), meanblob.width(), CV_32FC1, data);
        channels.push_back(channel);
        data += meanblob.height() * meanblob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat merged;
    cv::merge(channels, merged);
    cv::Scalar channel_mean = cv::mean(merged);
    std::cerr << channel_mean[0] << ' ' << channel_mean[1] << ' ' << channel_mean[2] << std::endl;
}
