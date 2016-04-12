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
#ifdef USE_CAFFE
        static Model *create_caffe (fs::path const &, int);
#endif
#ifdef USE_MXNET
        static Model *create_mxnet (fs::path const &, int);
#endif
#ifdef USE_PYTHON
        static Model *create_python (fs::path const &, int);
#endif
        virtual void apply (cv::Mat const &image, vector<float> *ft) {
            apply(vector<cv::Mat>{image}, ft);
        }
        virtual void apply (vector<cv::Mat> const &, vector<float> *) = 0;
        Model () {
            means[0] = means[1] = means[2] = 0;
        }
        virtual ~Model ();
    };

    class Tiler {
        struct Tiling {
            vector<cv::Rect> tiles;
            cv::Size size;

            int add2 (cv::Size sz, int x,
                      int top1, int top2, int bottom) {
                // top1 > top2
                tiles.emplace_back(x, 0, sz.width * top1/bottom,
                                         sz.height * top1/bottom);
                int rx = x + tiles.back().width;
                tiles.emplace_back(x, tiles.back().height,
                                         sz.width * top2/bottom,
                                         sz.height * top2/bottom);
                return rx;
            }

            Tiling (cv::Size sz) {
                int x = 0;
                tiles.emplace_back(0, 0, sz.width, sz.height);
                x += tiles.back().width;
                x = add2(sz, x, 3, 2, 5);
                x = add2(sz, x, 3, 1, 4);
                x = add2(sz, x, 2, 1, 3);
                size = cv::Size(x, sz.height);
            }
        };
    public:
        void forward (cv::Mat input, cv::Mat *tiled) {
            Tiling tiling(input.size());
            cv::Mat all(tiling.size, input.type(), cv::Scalar(0,0,0));
            input.copyTo(all(tiling.tiles[0]));
            for (unsigned i = 0; i < tiling.tiles.size(); ++i) {
                cv::resize(input, all(tiling.tiles[i]), tiling.tiles[i].size());
            }
            *tiled = all;
        }
        void backward (cv::Mat input, cv::Mat tiled, vector<cv::Mat> *) {
        }
    };
}

