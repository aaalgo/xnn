#include <queue>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "xnn.h"

namespace xnn {

int Model::mode = 0;

void Model::set_mode (int m) {
#ifndef CPU_ONLY
    mode = m;
#endif
}

float *Model::preprocess (cv::Mat const &image,
                          float *buffer, bool rgb) const {

    BOOST_VERIFY(image.data);
    BOOST_VERIFY(image.total());
    cv::Mat tmp;
    // convert color space
    if (image.channels() != channels()) {
        if (image.channels() == 3 && channels() == 1) {
            cv::cvtColor(image, tmp, CV_BGR2GRAY);
        }
        else if (image.channels() == 4 && channels() == 1) {
            cv::cvtColor(image, tmp, CV_BGRA2GRAY);
        }
        else if (image.channels() == 4 && channels() == 3) {
            cv::cvtColor(image, tmp, CV_BGRA2BGR);
        }
        else if (image.channels() == 1 && channels() == 3) {
            cv::cvtColor(image, tmp, CV_GRAY2BGR);
        }
        else {
            throw 0;
        }
    }
    else {
        tmp = image;
    }

    // check resize
    if ((shape[2] > 1)) {   // shape is fixed
        cv::Size sz(shape[3], shape[2]);
        if (sz != tmp.size()) {
            cv::resize(tmp, tmp, sz);
        }
    }

    int type = CV_32FC(channels());
    if (tmp.type() != type) {
        cv::Mat x;
        tmp.convertTo(x, type);
        tmp = x;
    }
    float *ptr_b = buffer;
    float *ptr_g = buffer;
    float *ptr_r = buffer;
    if (rgb) {
        ptr_g += tmp.total();
        ptr_b += 2 * tmp.total();
    }
    else if (channels() > 1) {
        ptr_g += tmp.total();
        ptr_r += 2 * tmp.total();
    }
    CHECK(tmp.elemSize() == channels() * sizeof(float));
    int off = 0;
    for (int i = 0; i < tmp.rows; ++i) {
        float const *line = tmp.ptr<float const>(i);
        for (int j = 0; j < tmp.cols; ++j) {
            if (channels() > 1) {
                float b = *line++;
                float g = *line++;
                b -= means[2];
                g -= means[1];
                ptr_b[off] = b;
                ptr_g[off] = g;
            }
            float r = *line++;
            r -= means[0];
            ptr_r[off] = r;
            ++off;
        }
    }
    CHECK(off == tmp.total());
    if (channels() == 1) return buffer + tmp.total();
    return buffer + 3 * tmp.total();
}

Model::~Model () {
}


Model *Model::create (fs::path const &dir, int batch) {
#ifdef USE_CAFFE
    if (fs::exists(dir / "caffe.model")) return create_caffe(dir, batch);
#endif
#ifdef USE_PYTHON
    if (fs::exists(dir / "model.py")) return create_python(dir, batch);
#endif
#ifdef USE_MXNET
    return create_mxnet(dir, batch);
#endif
    return nullptr;
}

}
