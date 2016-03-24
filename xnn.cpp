#include <queue>
#include <boost/assert.hpp>
#include <glog/logging.h>
#include "xnn.h"

namespace xnn {

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
    else {
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
    if (fs::exists(dir / "caffe.model")) return create_caffe(dir, batch);
    return create_mxnet(dir, batch);
}

static int conn_comp (cv::Mat *mat, cv::Mat const &weight, vector<BBox> *cnt) {
    // return # components
    CHECK(mat->type() == CV_32F);
    CHECK(mat->isContinuous());
    CHECK(weight.type() == CV_32F);
    CHECK(mat->size() == weight.size());
    cv::Mat out(mat->size(), CV_8UC1, cv::Scalar(0));
    CHECK(out.isContinuous());
    float const *ip = mat->ptr<float>(0);
    uint8_t *op = out.ptr<uint8_t>(0);
    float const *wp = weight.ptr<float const>(0);

    int c = 0;
    int o = 0;
    if (cnt) cnt->clear();
    for (int y = 0; y < mat->rows; ++y)
    for (int x = 0; x < mat->cols; ++x) {
        do {
            if (op[o]) break;
            if (ip[o] == 0) break;
            // find a new component
            ++c;
            std::queue<int> todo;
            op[o] = c;
            todo.push(o);
            int minx = mat->cols;
            int maxx = -1;
            int miny = mat->rows;
            int maxy = -1;
            while (!todo.empty()) {
                int t = todo.front();
                todo.pop();
                // find neighbors of t and add
                int tx = t % mat->cols;
                int ty = t / mat->cols;
                minx = std::min(tx, minx);
                miny = std::min(ty, miny);
                maxx = std::max(tx, maxx);
                maxy = std::max(ty, maxy);
                for (int ny = std::max(0, ty-1); ny <= std::min(mat->rows-1,ty+1); ++ny) 
                for (int nx = std::max(0, tx-1); nx <= std::min(mat->cols-1,tx+1); ++nx) {
                    // (ny, ix) is connected
                    int no = t + (ny-ty) * mat->cols + (nx-tx);
                    if (op[no]) continue;
                    if (ip[no] == 0) continue;
                    op[no] = c;
                    todo.push(no);
                }
            }
            if (cnt) {
                BBox comp;
                comp.box = cv::Rect(minx, miny, maxx - minx + 1, maxy - miny + 1);
                cnt->push_back(comp);
            }
        } while (false);
        ++o;
    }
    *mat = out;
    CHECK(c == cnt->size());
    return c;
}

template <typename I>
void bound (I begin, I end, int *b, float margin) {
    float cc = 0;
    I it = begin;
    while (it < end) {
        cc += *it;
        if (cc >= margin) break;
        ++it;
    }
    *b = it - begin;
}

static inline void bound (vector<float> const &v, int *x, int *w, float margin) {
    int b, e;
    bound(v.begin(), v.end(), &b, margin);
    bound(v.rbegin(), v.rend(), &e, margin);
    e = v.size() - e;
    if (e < b) e = b;
    *x = b;
    *w = e - b;
}

float accumulate (cv::Mat const &image, vector<float> *pX, vector<float> *pY) {
    vector<float> X(image.cols, 0);
    vector<float> Y(image.rows, 0);
    float total = 0;
    CHECK(image.type() == CV_32F);
    for (int y = 0; y < image.rows; ++y) {
        float const *row = image.ptr<float const>(y);
        for (int x = 0; x < image.cols; ++x) {
            float v = row[x];
            X[x] += v;
            Y[y] += v;
            total += v;
        }
    }
    pX->swap(X);
    pY->swap(Y);
    return total;
}

static inline void bound (cv::Mat const &image, cv::Rect *rect, float th) {
    vector<float> X;
    vector<float> Y;
    float total = accumulate(image, &X, &Y);
    float margin = total * (1.0 - th) / 2;
    bound(X, &rect->x, &rect->width, margin);
    bound(Y, &rect->y, &rect->height, margin);
}

void BBoxDetector::apply (cv::Mat prob, vector<BBox> *boxes) {
    boxes->clear();
    cv::Mat conn;
    cv::threshold(prob, conn, prob_th, 1, cv::THRESH_BINARY);
    conn_comp(&conn, prob, boxes);
    for (BBox &bbox: *boxes) {
        bbox.prob = prob(bbox.box);
        if (bound_th < 1) {
            cv::Rect kbox;
            bound(bbox.prob, &kbox, bound_th);
            kbox += bbox.box.tl();
            bbox.box = kbox;
            bbox.prob = prob(bbox.box);
        }
    }
}


}
