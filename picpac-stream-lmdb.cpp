#include <boost/filesystem/fstream.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/lexical_cast.hpp>
#include <caffe/proto/caffe.pb.h>
#include <caffe/util/db.hpp>
#include <caffe/util/io.hpp>
#include "picpac-cv.h"

using namespace std;
using namespace picpac;
using boost::scoped_ptr;

string backend("lmdb");

int main(int argc, char const* argv[]) {
    BatchImageStream::Config config;
    unsigned max;
    fs::path input_path;
    fs::path output_path;

    namespace po = boost::program_options;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message.")
        ("max", po::value(&max)->default_value(100), "")
        ("input", po::value(&input_path), "")
        ("output", po::value(&output_path), "")
        ;
#define PICPAC_CONFIG_UPDATE(C,p) desc.add_options()(#p, po::value(&C.p)->default_value(C.p), "")
    PICPAC_CONFIG_UPDATE_ALL(config);
#undef PICPAC_CONFIG_UPDATE

    po::positional_options_description p;
    p.add("input", 1);
    p.add("output", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help") || input_path.empty() || output_path.empty()) {
        cout << "Usage:" << endl;
        cout << "\tpicpac-stat ... <db>" << endl;
        cout << desc;
        cout << endl;
        return 0;
    }
    ImageStream db(input_path, config);
    scoped_ptr<caffe::db::DB> image_db(caffe::db::GetDB(backend));
    image_db->Open(output_path.native(), caffe::db::NEW);
    scoped_ptr<caffe::db::Transaction> image_txn(image_db->NewTransaction());
    int c = 0;
    for (unsigned i = 0; i < max; ++i) {
        try {
            ImageStream::Value v(db.next());
            CHECK(v.image.total() > 0);
            caffe::Datum datum;
            caffe::CVMatToDatum(v.image, &datum);
            datum.set_label(v.label);
            string key = lexical_cast<string>(c) , value;
            CHECK(datum.SerializeToString(&value));
            image_txn->Put(key, value);
            ++c;
        }
        catch (EoS const &) {
            break;
        }
    }
    image_txn->Commit();

    return 0;
}

