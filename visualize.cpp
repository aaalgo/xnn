#include <memory>
#include <boost/assert.hpp>
#include <iostream>
#include <boost/program_options.hpp>
#include <glog/logging.h>
#include "xnn.h"


using namespace std;
using namespace boost;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    string model;
    string ipath;
    string opath;
    float b_th;
    float b_keep;
    int mode;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model", po::value(&model)->default_value("model"), "")
    ("input", po::value(&ipath), "")
    ("output", po::value(&opath), "")
    ("th", po::value(&b_th)->default_value(0.05), "")
    ("keep", po::value(&b_keep)->default_value(0.95), "")
    ("mode", po::value(&mode)->default_value(0), "")
    ;


    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || ipath.empty()) {
        cerr << desc;
        return 1;
    }

    xnn::Model::set_mode(mode);
    unique_ptr<xnn::Model> det(xnn::Model::create(model));
    CHECK(det->fcn());
    cv::Mat ret;
    cv::Mat input = cv::imread(ipath, CV_LOAD_IMAGE_COLOR);
    BOOST_VERIFY(input.data);
    vector<float> resp;
    det->apply(vector<cv::Mat>{input}, &resp);
    CHECK(resp.size() == input.total() * 2) << resp.size() << ' ' << input.total();
    for (auto &v: resp) {
        v = 1.0 - v;
    }
    cv::Mat fl;
    input.convertTo(fl, CV_32FC3);
    cv::Mat prob(input.size(), CV_32F, &resp[0]);
    vector<xnn::BBox> boxes;
    xnn::BBoxDetector bdet(b_th, b_keep);
    bdet.apply(prob, &boxes);

    vector<cv::Mat> chs{prob, prob, prob};
    cv::Mat prob3d;
    cv::merge(&chs[0], 3, prob3d);
    cv::Mat mask = fl.mul(prob3d);
    prob3d *= 255;
    for (auto const &box: boxes) {
        cv::rectangle(prob3d, box.box, cv::Scalar(0, 0, 0xFF), 2);
        cv::rectangle(mask, box.box, cv::Scalar(0, 0, 0xFF), 2);
        cv::rectangle(fl, box.box, cv::Scalar(0, 0, 0xFF), 2);
    }
    cv::hconcat(mask, prob3d, mask);
    cv::hconcat(mask, fl, fl);
    cv::imwrite(opath, fl);
    return 0;
}

