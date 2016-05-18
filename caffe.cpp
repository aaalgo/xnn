#include <vector>
#include <caffe/caffe.hpp>
#include <boost/shared_ptr.hpp>
#include "xnn.h"

namespace xnn {

using namespace caffe;
using std::vector;
using boost::shared_ptr;

class CaffeSetMode {
public:
    CaffeSetMode (int mode) {
        if (mode == 0) {
            Caffe::set_mode(Caffe::CPU);
        }
        else {
            Caffe::set_mode(Caffe::GPU);
        }
    }
};

class CaffeModel: public Model, CaffeSetMode {
    Net<float> net;
    Blob<float> *input_blob;
    vector<shared_ptr<Blob<float>>> output_blobs;
public:
    CaffeModel (fs::path const& dir, int batch)
        : CaffeSetMode(mode),
        net((dir/"caffe.model").native(), TEST)
    {
        BOOST_VERIFY(batch >= 1);
        CHECK_EQ(net.num_inputs(), 1) << "Network should have exactly one input: " << net.num_inputs();
        input_blob = net.input_blobs()[0];
        shape[0] = batch;
        shape[1] = input_blob->shape(1);
        CHECK(shape[1] == 3 || shape[1] == 1)
            << "Input layer should have 1 or 3 channels." << shape[1];
        net.CopyTrainedLayersFrom((dir/"caffe.params").native());
        // resize to required batch size
        shape[2] = input_blob->shape(2);
        shape[3] = input_blob->shape(3);
        input_blob->Reshape(shape[0], shape[1], shape[2], shape[3]);
        net.Reshape();
        // set mean file
        means[0] = means[1] = means[2] = 0;
        fs::path mean_file = dir / "caffe.mean";
        fs::ifstream test(mean_file);
        if (test) {
            BlobProto blob_proto;
            // check old format
            if (ReadProtoFromBinaryFile(mean_file.native(), &blob_proto)) {
                /* Convert from BlobProto to Blob<float> */
                Blob<float> meanblob;
                meanblob.FromProto(blob_proto);
                CHECK_EQ(meanblob.channels(), channels())
                    << "Number of channels of mean file doesn't match input layer.";
                /* The format of the mean file is planar 32-bit float BGR or grayscale. */
                vector<cv::Mat> mats;
                float* data = meanblob.mutable_cpu_data();
                for (int i = 0; i < channels(); ++i) {
                    /* Extract an individual channel. */
                    cv::Mat channel(meanblob.height(), meanblob.width(), CV_32FC1, data);
                    mats.push_back(channel);
                    data += meanblob.height() * meanblob.width();
                }
                /* Merge the separate channels into a single image. */
                cv::Mat merged;
                cv::merge(mats, merged);
                cv::Scalar channel_mean = cv::mean(merged);
                //mean = cv::Mat(input_height, input_width, merged.type(), channel_mean);
                means[0] = means[1] = means[2] = channel_mean[0];
                if (channels() > 1) {
                    means[1] = channel_mean[1];
                    means[2] = channel_mean[2];
                }   
            }
            // if not proto format, then the mean file is just a bunch of textual numbers
            else {
                test >> means[0];
                means[1] = means[2] = means[0];
                test >> means[1];
                test >> means[2];
            }
        }
        {
            fs::ifstream is(dir/"blobs");
            string blob;
            CHECK(is) << "cannot open blobs file.";
            while (is >> blob) {
                output_blobs.push_back(net.blob_by_name(blob));
            }
        }
    }

    virtual void apply (vector<cv::Mat> const &images, vector<float> *ft) {
        int batch = shape[0];
        CHECK(!images.empty()) << "must input >= 1 images";
        CHECK(images.size() <= batch) << "Too many input images.";
        if (fcn()) {    // for FCN, we need to resize network according to image size
            cv::Size sz = images[0].size();
            for (unsigned i = 1; i < images.size(); ++i) {
                CHECK(images[i].size() == sz) << "all images must be the same size";
            }
            int input_height = input_blob->shape(2);
            int input_width = input_blob->shape(3);
            if ((input_width != sz.width)
                    || (input_height != sz.height)) {
                input_blob->Reshape(shape[0], shape[1], sz.height, sz.width);
                net.Reshape();
            }
        }
        float *input_data = input_blob->mutable_cpu_data();
        float *e = preprocess(images, input_data);
        CHECK(e -input_data <= input_blob->count());
        net.ForwardPrefilled();

        // compute output dimension
        int dim = 0;
        for (auto const &b: output_blobs) {
            int d = b->count() / batch;
            LOG(INFO) << "output: " << b->shape_string();
            dim += d;
        }
        LOG(INFO) << "output size " << images.size() << " x " << dim;
        ft->resize(images.size() * dim);    // total output size
        int off = 0;
        for (auto const &b: output_blobs) {
            int blob_dim = b->count() / batch;
            float const *from_begin = b->cpu_data();
            for (int i = 0; i < images.size(); ++i) {
                float const *from_end = from_begin + blob_dim;
                std::copy(from_begin, from_end, &ft->at(i * dim + off));
                from_begin = from_end;
            }
            off += blob_dim;
        } 
        CHECK(off == dim);
    }
};

Model *Model::create_caffe (fs::path const &dir, int batch) {
    return new CaffeModel(dir, batch);
}

}


