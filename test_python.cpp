#include <malloc.h>
#include <iostream>
#include <boost/timer/timer.hpp>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#undef timer
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "xnn.h"

using namespace std;
using namespace boost;

int main( int argc, char ** argv ) {
    namespace po = boost::program_options; 
    static int batch;
    static int loop;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message.")
    ("batch", po::value(&batch)->default_value(64), "")
    ("loop", po::value(&loop)->default_value(64), "")
    ;

    po::positional_options_description p;

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
                     options(desc).positional(p).run(), vm);
    po::notify(vm); 

    if (vm.count("help")) {
        cerr << desc;
        return 1;
    }

    cv::Mat image = cv::imread("a.jpg", CV_LOAD_IMAGE_GRAYSCALE);
    cv::resize(image, image, cv::Size(256,256));
    {
        cv::Mat tmp;
        image.convertTo(tmp, CV_32F);
        image = tmp;

    }
    CHECK(image.channels() == 1);
    vector<float> output;
    xnn::Model *model = xnn::Model::create_python("models/wv1", batch);
    CHECK(model);
    {
        cerr << "Ramp up with initial batch";
        {
            vector<cv::Mat> images(batch, image);
            model->apply(images, &output);
        }
        boost::timer::auto_cpu_timer t;
        progress_display progress(loop, cerr);
        struct mallinfo mbefore = mallinfo();
        for (int i = 0; i < loop; ++i) {
            vector<cv::Mat> images(batch, image);
            model->apply(images, &output);
            ++progress;
        }
        struct mallinfo mafter = mallinfo();
        cerr << "Finished processing " << batch * loop << " images." << endl;
        cerr << "leak: " << mafter.uordblks - mbefore.uordblks << endl;
    }
    CHECK(output.size() > image.total());
    cv::Mat label(image.rows, image.cols, CV_32F, &output[0]);
    label *= 255;
    cv::imwrite("xxx.jpg", label);
    delete model;
    return 0;
}

