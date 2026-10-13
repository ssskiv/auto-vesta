#include "vesta_neural_controller/onnx_policy.hpp"

#include <array>
#include <vector>

#include "vesta_neural_controller/observation.hpp"

#ifdef VESTA_HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace vesta_neural_controller
{

#ifdef VESTA_HAVE_ONNXRUNTIME

struct OnnxPolicy::Impl
{
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vesta_neural_controller"};
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
};

OnnxPolicy::OnnxPolicy()
: impl_(std::make_unique<Impl>()) {}

OnnxPolicy::~OnnxPolicy() = default;

bool OnnxPolicy::load(const std::string & path, std::string & error)
{
  try {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, path.c_str(), options);

    auto check = [&](std::size_t i, const std::vector<int64_t> & want, const char * name) {
        auto shape = impl_->session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != want.size()) {
          error = std::string("input ") + name + " has the wrong rank";
          return false;
        }
        for (std::size_t d = 1; d < shape.size(); ++d) {  // dim 0 is the batch
          if (shape[d] != want[d]) {
            error = std::string("input ") + name +
              " does not match observation.hpp (PATCH_SIZE / VEC_SIZE changed?)";
            return false;
          }
        }
        return true;
      };
    if (impl_->session->GetInputCount() != 2 ||
      !check(0, {-1, 1, obs::PATCH_SIZE, obs::PATCH_SIZE}, "costmap") ||
      !check(1, {-1, obs::VEC_SIZE}, "vec"))
    {
      if (error.empty()) {
        error = "expected inputs (costmap, vec)";
      }
      impl_->session.reset();
      return false;
    }
  } catch (const Ort::Exception & e) {
    error = e.what();
    impl_->session.reset();
    return false;
  }
  return true;
}

bool OnnxPolicy::loaded() const
{
  return impl_->session != nullptr;
}

bool OnnxPolicy::infer(const float * costmap, const float * vec, float * action, std::string & error)
{
  if (!impl_->session) {
    error = "no model loaded";
    return false;
  }
  try {
    const std::array<int64_t, 4> cm_shape{1, 1, obs::PATCH_SIZE, obs::PATCH_SIZE};
    const std::array<int64_t, 2> vec_shape{1, obs::VEC_SIZE};
    std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<float>(
        impl_->memory, const_cast<float *>(costmap), obs::PATCH_SIZE * obs::PATCH_SIZE,
        cm_shape.data(), cm_shape.size()),
      Ort::Value::CreateTensor<float>(
        impl_->memory, const_cast<float *>(vec), obs::VEC_SIZE, vec_shape.data(), vec_shape.size())};
    const char * in_names[] = {"costmap", "vec"};
    const char * out_names[] = {"action"};
    auto out = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 2, out_names, 1);
    const float * a = out[0].GetTensorData<float>();
    action[0] = a[0];
    action[1] = a[1];
  } catch (const Ort::Exception & e) {
    error = e.what();
    return false;
  }
  return true;
}

#else  // built without onnxruntime

struct OnnxPolicy::Impl {};

OnnxPolicy::OnnxPolicy()
: impl_(std::make_unique<Impl>()) {}

OnnxPolicy::~OnnxPolicy() = default;

bool OnnxPolicy::load(const std::string &, std::string & error)
{
  error = "vesta_neural_controller was built without onnxruntime";
  return false;
}

bool OnnxPolicy::loaded() const {return false;}

bool OnnxPolicy::infer(const float *, const float *, float *, std::string & error)
{
  error = "vesta_neural_controller was built without onnxruntime";
  return false;
}

#endif

}  // namespace vesta_neural_controller
