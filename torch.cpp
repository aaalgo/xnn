#include <string>
#include <vector>
#include <json11.hpp>
#include <glog/logging.h>
#include <lua.hpp>
#include <luaT.h>
#include <TH/TH.h>
#include "xnn.h"

/* How to use Torch model

To build, link against -lTH -lluaT -lluajit -llapack -lopenblas
The TH library is here: https://github.com/torch/TH
You should get the rest when you installed torch.

Prepare a "lua_config" file containing the following content;
this will be called to initialize the torch environment and load
the model.

  require 'torch'
  require 'nn'
  require 'dpnn'

  io.stdout:setvbuf 'no'
  torch.setdefaulttensortype('torch.FloatTensor')

  require 'torch'
  require 'nn'
  require 'dpnn'

  io.stdout:setvbuf 'no'
  torch.setdefaulttensortype('torch.FloatTensor')

  torch.setnumthreads(1)    -- In server we usually want to run single-threaded
                            -- so the server itself can run multiple threads
  net = nil·

  function setup (model)    -- this function will be called by C++ constructor
      net = torch.load(model)
      net:evaluate()
  end

  function apply (input)
      return net:forward(input)
  end
*/

namespace xnn {

using namespace json11;
using std::string;
using std::vector;

class TorchModel: public Model {
    lua_State *state;
public:
    TorchModel (fs::path const &lua_config, fs::path const& model_path, int batch, int H, int W, int C)
        : state(luaL_newstate())
    // we are taking in input tensor shape here
    // will have to change code to get the shape from the lua_config
    {
        BOOST_VERIFY(batch >= 1);
        if (!state) throw std::runtime_error("failed to initialize Lua");
        luaL_openlibs(state);
        int result = luaL_loadfile(state, lua_config.c_str());
        if (result != 0) throw std::runtime_error("failed to load openface server");
        result = lua_pcall(state, 0, LUA_MULTRET, 0);
        if (result != 0) throw std::runtime_error("failed to run openface server");
        // call lua's setup with model path
        lua_getglobal(state, "setup");
        lua_pushstring(state, model_path.c_str());
        if (lua_pcall(state, 1, 0, 0) != 0) {
            throw std::runtime_error("fail to extract");
        }

        shape[0] = batch;
        shape[1] = C;
        shape[2] = H;
        shape[3] = W;
    }

    ~TorchModel () {
        lua_close(state);
    }

    // The images will form a batch.  the elements of the output tensor
    // will be densely added to the ft vector. 
    virtual void apply (vector<cv::Mat> const &images, vector<float> *ft) {
        // lua will take care of memory release
        THFloatTensor *itensor = THFloatTensor_newWithSize4d(shape[0], shape[1], shape[2], shape[3]);
        if (!THFloatTensor_isContiguous(itensor)) {
            throw std::runtime_error("Torch tensor is not contiguous.");
        }
        float *origin = THFloatTensor_data(itensor);
        // pre-process images and load data to buffer
        float *e = preprocess(images, &origin);

        // call "apply" in lua
        lua_getglobal(state, "apply");
        luaT_pushudata(state, itensor, "torch.FloatTensor");
        if (lua_pcall(state, 1, 1, 0) != 0) {
            throw std::runtime_error("fail to extract");
        }
        THFloatTensor const*otensor = reinterpret_cast<THFloatTensor const*>(luaT_toudata(state, -1, "torch.FloatTensor"));
        if (!THFloatTensor_isContiguous(otensor)) {
            throw std::runtime_error("Torch output tensor is not contiguous.");
        }
        long dim = THFloatTensor_nDimension(otensor);
        // get output size
        size_t sz = 1;
        for (long i = 0; i < dim; ++i) {
            sz *= THFloatTensor_size(otensor, i);
        }

        float const *ptr = THFloatTensor_data(otensor);
        ft->resize(sz);
        for (long i = 0; i < sz; ++i) {
            ft->at(i) = ptr[i];
        }
        lua_pop(state, 1);
    }
};

Model *create_torch (fs::path const &lua_config, int batch, int H, int W, int C) {
    return new TorchModel(lua_config, batch, H, W, C);
}

}


