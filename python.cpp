#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <string>
#include <vector>
#include <glog/logging.h>
#include "xnn.h"

namespace xnn {

using std::string;
using std::vector;

static bool python_initialized = false;

void check_import_array () {
    import_array();
}

PyObject *module_call (PyObject *module, char const *name, PyObject *args) {
    PyObject *pFunc = PyObject_GetAttrString(module, name);
    CHECK(pFunc);
    //cerr << "Loaded: " << pFunc << endl;
    CHECK(PyCallable_Check(pFunc));
    /* pFunc is a new reference */
    PyObject *ret = PyObject_CallObject(pFunc, args);
    Py_DECREF(pFunc);
    Py_DECREF(args);
    return ret;
}

class PythonModel: public Model {
    PyObject *module;
    PyObject *predict;
    PyObject *input;
    vector<float> image_data;
public:
    PythonModel (fs::path const& dir, int batch)
    {
        CHECK(!python_initialized);
        python_initialized = true;
        // initialize python
        Py_Initialize();
        check_import_array();
        {  // add search path
            PyObject* sysPath = PySys_GetObject((char*)"path");
            BOOST_VERIFY(sysPath);
            PyObject* cwd = PyString_FromString(dir.native().c_str());
            BOOST_VERIFY(cwd);
            PyList_Append(sysPath, cwd);
            Py_DECREF(cwd);
        }
        BOOST_VERIFY(batch >= 1);

        module = PyImport_ImportModule("model");
        CHECK(module);
        PyObject *accept_shape = module_call(module, "shape", Py_BuildValue("()"));
        CHECK(accept_shape);
        PyArg_ParseTuple(accept_shape, "iiii", &shape[0], &shape[1], &shape[2], &shape[3]);
        Py_DECREF(accept_shape);
        shape[0] = batch;
        predict = module_call(module, "load",
                Py_BuildValue("((iiii))", shape[0], shape[1], shape[2], shape[3]));
        CHECK(predict);
        CHECK(PyCallable_Check(predict));
        npy_intp dims[] = {shape[0], shape[1], shape[2], shape[3]};
        input = PyArray_SimpleNew(4, dims, NPY_FLOAT);
        CHECK(input);
        CHECK(!fcn());
    }

    ~PythonModel () {
        Py_DECREF(input);
        Py_DECREF(predict);
        Py_DECREF(module);
        Py_Finalize();
    }

    virtual void apply (vector<cv::Mat> const &images, vector<float> *ft) {
        // Just a big enough memory 1000x1000x3
        int batch = shape[0];
        CHECK(!images.empty()) << "must input >= 1 images";
        CHECK(images.size() <= batch) << "Too many input images.";
#if 0
        if (fcn()) {    // for FCN, we need to resize network according to image size
            cv::Size sz = images[0].size();
            for (unsigned i = 1; i < images.size(); ++i) {
                CHECK(images[i].size() == sz) << "all images must be the same size";
            }
            int bufsz = image_buffer_size(images[0]);
            image_data.resize(bufsz * images.size());
        }
#endif
        float *e = preprocess(images, reinterpret_cast<float *>(PyArray_DATA(input)));
        PyObject *tuple = Py_BuildValue("(O)", input);
        PyArrayObject *output = (PyArrayObject *)PyObject_CallObject(predict, tuple);
        CHECK(output);
        Py_DECREF(tuple);
        CHECK(PyArray_ITEMSIZE(output) == sizeof(float));
        float const *from = reinterpret_cast<float const *>(PyArray_DATA(output));
        int sz = PyArray_SIZE(output);
        ft->resize(sz);
        std::copy(from, from + sz, ft->begin());
        Py_DECREF(output);
    }
};

Model *Model::create_python (fs::path const &dir, int batch) {
    return new PythonModel(dir, batch);
}

}


