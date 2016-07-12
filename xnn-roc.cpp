#include <memory>
#include <boost/program_options.hpp>
#include "xnn.h"
#include "picpac-cv.h"

using namespace std;
using namespace boost;
using namespace xnn;

/*
 * precision =  tp / (tp + fp)
 * recall = tp / p
 */

// thresholding pixels by 0/N, 1/N, ...., N/N, and produce a curve
// of x: (tp + fp)/total, y: tp/(all_positive)
void roc (unique_ptr<Model> &model, picpac::ImageStream::Value &v, bool tile,
        int N, vector<pair<double, double>> *curve) {
    vector<float> resp; // prob response
    model->apply(v.image, &resp);
    CHECK(v.annotation.type() == CV_8UC1);
    CHECK(resp.size() % v.image.total() == 0);
    CHECK(v.image.size() == v.annotation.size());
    // response is   N prob of bg, and then N prob of fg == (1--bg)
    float *p = &resp[0];
    // pair P(fg), label
    vector<pair<float, int>> all;
    for (int i = 0; i < v.image.rows; ++i) {
        uint8_t const *row = v.annotation.ptr<uint8_t const>(i);
        for (int j = 0; j < v.image.cols; ++j) {
            all.push_back(std::make_pair(*p, row[j]));
            ++p;
        }
    }
    sort(all.begin(), all.end());
    int tp = 0;
    int off = 0;
    curve->clear();
    for (int i = 0; i <= N; ++i) {
        while ((off < all.size()) && (all[off].first * N <= i)) {
            if (all[off].second) {
                ++tp;
            }
            ++off;
        }
        curve->emplace_back(off, tp);
    }
    /*
    double area = 0;
    unsigned total = 0;
    for (auto const &p: all) {
        unsigned total2 = total + p.second;
        area += total + total2;
        total = total2;
    }
    //cout << total << '\t' << all.size() << endl;
    area /= 2;
    area /= total;
    area /= all.size();
    */
    for (auto &p: *curve) {
        p.first /= all.size();
        p.second /= tp;
    }
}

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    picpac::BatchImageStream::Config config;
    config.loop = false;
    config.shuffle = true;
    config.stratify = true;
    config.split_negate = true;
    config.anno_type = CV_8UC1;
    config.anno_color1 = 1;
    config.anno_thickness = -1; // fill

    fs::path model_dir;
    fs::path db_path;
    int mode;
    int batch;
    int max_size;
    unsigned N;
    bool tile;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model", po::value(&model_dir), "")
    ("db", po::value(&db_path), "")
    ("mode", po::value(&mode)->default_value(0), "0 for CPU or 1 for GPU")
    ("max", po::value(&config.max_size)->default_value(-1), "")
    ("max-size", po::value(&config.max_size)->default_value(800), "")
    /*
    ("split,s", po::value(&config.split)->default_value(5), "")
    ("fold,f", po::value(&config.split_fold)->default_value(0), "")
    ("stratify", po::value(&config.stratify)->default_value(true), "")
    ("shuffle", po::value(&config.shuffle)->default_value(true), "")
    ("annotate", po::value(&config.annotate)->default_value("none"), "none for classification")
    ("negate", po::value(&config.split_negate)->default_value(true), "")
    */
    ("level", po::value(&FLAGS_minloglevel)->default_value(1),"")
    (",N", po::value(&N)->default_value(1000), "")
    //("batch", po::value(&batch)->default_value(1), "")
    ("tile", "")
    ;
#define PICPAC_CONFIG_UPDATE(C,p) desc.add_options()(#p, po::value(&C.p)->default_value(C.p), "")
    PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE

    po::positional_options_description p;
    p.add("model", 1);
    p.add("db", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || model_dir.empty() || db_path.empty()) {
        cerr << desc;
        return 1;
    }
    FLAGS_logtostderr = 1;
    tile = vm.count("tile") > 0;
    google::InitGoogleLogging(argv[0]);
    Model::set_mode(mode);
    unique_ptr<Model> model(Model::create(model_dir, config.batch));
    picpac::ImageStream db(db_path, config);
    if (config.annotate == "none") {
        /*
        int total = 0;
        int correct = 0;
        */
        map<int, pair<int, int>> cnt;
        for (;;) {
            vector<cv::Mat> images;
            vector<int> labels;
            for (unsigned i = 0; i < config.batch; ++i) {
                try {
                    picpac::ImageStream::Value v(db.next());
                    images.push_back(v.image);
                    CHECK(v.image.total() > 0);
                    unsigned l = v.label;
                    CHECK(l == v.label);
                    labels.push_back(l);
                }
                catch (picpac::EoS const &) {
                    break;
                }
            }
            if (images.empty()) break;
            vector<float> resp; // prob response
            model->apply(images, &resp);
            float const *off = &resp[0];
            CHECK(resp.size() % images.size() == 0);
            size_t nc = resp.size() / images.size();
            for (unsigned l: labels) {
                CHECK(l < nc);
                bool ok = true;
                for (unsigned c = 0; c < nc; ++c) {
                    if (off[l] < off[c]) {
                        ok = false;
                        break;
                    }
                }
                auto &p = cnt[l];
                ++p.second;
                if (ok) ++p.first;
                off += nc;
            }
            if (images.size() < config.batch) break;
        }
        double sum = 0;
        for (auto const &p: cnt) {
            double r = 1.0 * p.second.first / p.second.second;
            sum += r;
            std::cout << p.first << ':' << r << '\t';
        }
        std::cout << sum / cnt.size() << std::endl;
    }
    else {
        int cnt = 0;
        vector<pair<double, double>> sum(N+1, std::make_pair(0,0));
        vector<pair<double, double>> curv;
        for (;;) {
            try {
                picpac::ImageStream::Value v(db.next());
                CHECK(v.image.total() > 0);
                CHECK(v.annotation.total() > 0);
                roc(model, v, tile, N, &curv);
                for (unsigned i = 0; i <= N; ++i) {
                    sum[i].first += curv[i].first;
                    sum[i].second += curv[i].second;
                }
                cnt += 1;
                //cout << a << endl;
            }
            catch (picpac::EoS const &) {
                break;
            }
        }
        for (auto const &p: sum) {
            cout << p.first/cnt << '\t' << p.second/cnt << endl;
        }
    }
    return 0;
}

