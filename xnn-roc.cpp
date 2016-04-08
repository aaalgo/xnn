#include <memory>
#include <boost/program_options.hpp>
#include "xnn.h"
#include "picpac-cv.h"

using namespace std;
using namespace boost;
using namespace xnn;

// thresholding pixels by 0/N, 1/N, ...., N/N, and produce a curve
// of x: (tp + fp)/total, y: tp/(all_positive)
void roc (unique_ptr<Model> &model, picpac::ImageStream::Value &v,
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
    picpac::ImageStream::Config config;
    config.loop = false;
    config.stratify = false;
    config.anno_type = CV_8UC1;
    config.anno_color1 = 1;
    config.anno_thickness = -1; // fill

    fs::path model_dir;
    fs::path db_path;
    int mode;
    unsigned N;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model", po::value(&model_dir), "")
    ("db", po::value(&db_path), "")
    ("mode", po::value(&mode)->default_value(0), "")
    ("max", po::value(&config.max_size)->default_value(-1), "")
    ("split,s", po::value(&config.split)->default_value(5), "")
    ("fold,f", po::value(&config.split_fold)->default_value(0), "")
    ("anno", po::value(&config.annotate)->default_value("json"), "")
    ("channels", po::value(&config.channels)->default_value(-1), "")
    ("negate", po::value(&config.split_negate)->default_value(true), "")
    ("level", po::value(&FLAGS_minloglevel)->default_value(1),"")
    (",N", po::value(&N)->default_value(1000), "")
    ;

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
    google::InitGoogleLogging(argv[0]);
    Model::set_mode(mode);
    unique_ptr<Model> model(Model::create(model_dir));
    picpac::ImageStream db(db_path, config);
    int cnt = 0;
    vector<pair<double, double>> sum(N+1, std::make_pair(0,0));
    vector<pair<double, double>> curv;
    for (;;) {
        try {
            picpac::ImageStream::Value v(db.next());
            roc(model, v, N, &curv);
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
    return 0;
}

