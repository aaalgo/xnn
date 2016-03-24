#include <string>
#include <vector>
#include <json11.hpp>
#include <mxnet/c_predict_api.h>
#include <glog/logging.h>
#include "xnn.h"

namespace xnn {

using namespace json11;
using std::string;
using std::vector;

static void readall (fs::path const &path, string *content) {
    fs::ifstream ifs(path, std::ios::in | std::ios::binary);
    CHECK(ifs);
    ifs.seekg(0, std::ios::end);
    size_t length = ifs.tellg();
    CHECK(ifs);
    content->resize(length);
    ifs.seekg(0, std::ios::beg);
    ifs.read(&content->at(0), content->size());
}

class MXNetModel: public Model {
    PredictorHandle out;
    vector<float> image_data;
public:
    MXNetModel (fs::path const& dir, int batch)
    {
        BOOST_VERIFY(batch >= 1);
        string symbol;
        string params;
        string meta;
        readall(dir/"mxnet.symbol", &symbol);
        readall(dir/"mxnet.params", &params);
        readall(dir/"mxnet.meta", &meta);
        string json_err;

        Json json = Json::parse(meta, json_err);
        CHECK(json_err.empty());

        int dev_type = 1;  // 1: cpu, 2: gpu
        int dev_id = 0;  // arbitrary.
        mx_uint num_input_nodes = 1;  // 1 for feedforward
        const char* input_key[1] = {"data"};
        const char** input_keys = input_key;

        int width = json["shape"].number_value();
        int height = width;
        int channels = json["channels"].number_value();
        if (json["mean"].is_array()) {
            int x = 0;
            for (auto v: json["mean"].array_items()) {
                means[x++] = v.number_value();
            }
            CHECK(x == 3);
        }
        else {
            means[0] = means[1] = means[2] = json["mean"].number_value();
        }

        shape[0] = batch;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;

        const mx_uint input_shape_indptr[2] = { 0, 4 };
        const mx_uint input_shape_data[4] = {batch,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(width),
                                        static_cast<mx_uint>(height) };
        // ( trained_width, trained_height, channel, num)
        MXPredCreate(symbol.c_str(),
                     &params[0],
                     params.size(),
                     dev_type,
                     dev_id,
                     num_input_nodes,
                     input_keys,
                     input_shape_indptr,
                     input_shape_data,
                     &out);
        image_data.resize(shape[0] * shape[1] * shape[2] * shape[3]);
    }

    ~MXNetModel () {
        MXPredFree(out);
    }

    virtual void apply (vector<cv::Mat> const &images, vector<float> *ft) {
        // Just a big enough memory 1000x1000x3
        int batch = shape[0];
        CHECK(!images.empty()) << "must input >= 1 images";
        CHECK(images.size() <= batch) << "Too many input images.";
        if (fcn()) {    // for FCN, we need to resize network according to image size
            cv::Size sz = images[0].size();
            for (unsigned i = 1; i < images.size(); ++i) {
                CHECK(images[i].size() == sz) << "all images must be the same size";
            }
            int bufsz = image_buffer_size(images[0]);
            image_data.resize(bufsz * images.size());
        }
        float *e = preprocess(images, &image_data[0], true);
        CHECK(e - &image_data[0] == image_data.size());
        //-- Set Input Image
        MXPredSetInput(out, "data", image_data.data(), image_data.size());
        //-- Do Predict Forward
        MXPredForward(out);

        mx_uint output_index = 0;
        mx_uint *shape = 0;
        mx_uint shape_len;

        //-- Get Output Result
        MXPredGetOutputShape(out, output_index, &shape, &shape_len);

        size_t size = 1;
        for (mx_uint i = 0; i < shape_len; ++i) size *= shape[i];

        ft->resize(size);
        MXPredGetOutput(out, output_index, &ft->at(0), size);
    }
};

Model *Model::create_mxnet (fs::path const &dir, int batch) {
    return new MXNetModel(dir, batch);
}

}


