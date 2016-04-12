#pragma once
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

namespace xnn {
    using std::array;
    using std::vector;
    namespace fs = boost::filesystem;

    class Model {
    protected:
        static int mode;        // 0: CPU, 1: GPU
        array<int, 4> shape;    // batch shape
                                // batch_size
                                // channel
                                // rows, -1 for FCN
                                // cols, -1 for FCN
        array<float, 3> means;  // pixel means, R, G, B
        // save data to buffer, return buffer + data_size
        // rgb = true: output is RGB, else output is BGR (input is always BGR)
        float *preprocess (cv::Mat const &, float *buffer, bool rgb) const;
        float *preprocess (vector<cv::Mat> const &images, float *buffer, bool rgb) const {
            for (auto const &image: images) {
                buffer = preprocess(image, buffer, rgb);
            }
            return buffer;
        }
        int image_buffer_size (cv::Mat const &image) {
            if (shape[2] > 1) {
                return shape[1] * shape[2] * shape[3];
            }
            return shape[1] * image.total();
        }
    public:
        static void set_mode (int m);
        bool fcn () const { return shape[2] <= 1; }
        int batch () const { return shape[0];}
        int channels () const { return shape[1];}
        static Model *create (fs::path const &, int = 1);
        static Model *create_caffe (fs::path const &, int);
        static Model *create_mxnet (fs::path const &, int);
        static Model *create_python (fs::path const &, int);
        virtual void apply (cv::Mat const &image, vector<float> *ft) {
            apply(vector<cv::Mat>{image}, ft);
        }
        virtual void apply (vector<cv::Mat> const &, vector<float> *) = 0;
        Model () {
            means[0] = means[1] = means[2] = 0;
        }
        virtual ~Model ();
    };
};

