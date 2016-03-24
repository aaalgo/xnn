#include <memory>
#include <boost/program_options.hpp>
#include "xnn.h"

using namespace std;
using namespace boost;
using namespace xnn;

int main(int argc, char **argv) {
    namespace po = boost::program_options; 
    fs::path model_dir;
    fs::path image_path;
    int mode;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("model", po::value(&model_dir), "")
    ("path", po::value(&image_path), "")
    ("mode", po::value(&mode)->default_value(0), "")
    ;

    po::positional_options_description p;
    p.add("model", 1);
    p.add("path", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help") || model_dir.empty() || image_path.empty()) {
        cerr << desc;
        return 1;
    }
    Model::set_mode(mode);
    unique_ptr<Model> model(Model::create(model_dir));
    cv::Mat image = cv::imread(image_path.native(), -1);
    vector<float> ft;
    model->apply(image, &ft);
    for (unsigned i = 0; i < ft.size(); ++i){
        cout << i << '\t' << ft[i] << endl;
    }
    return 0;
}

