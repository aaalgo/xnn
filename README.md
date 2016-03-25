XNN: A C++ Prediction API Library for Caffe, MXNET and Python
=============================================================

Author: Wei Dong (wdong@wdong.org)

# Usage
```
#include <xnn.h>

...

int batch = 1;
xnn::Model *model = xnn::Model::create("model dir", batch);

cv::Mat image = cv::read("some.jpg", -1);
vector<float> out;

model->apply(image, &out);
model->apply(vector<cv::Mat>{image}, &out); 
```

At most batch images can be passed in each invokation of Model::apply.
The returned vector will have the size of (# category * batch) or
(out image size * batch), depending on whether the model does
classification or segmentation.

# Building and Installation

The library depends on that Caffe, MXNet, Theano/Lasagne and other python
libraries are properly installed.  Use [these scripts](https://github.com/aaalgo/centos7-deep)
to install everything on a fresh CentOS 7 installation.


