#include "offroad_nav2/engine.hpp"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <future>
#include <stdexcept>

namespace offroad {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

struct Net {
  std::unique_ptr<Ort::Session> session;
  std::vector<int64_t> out_shape;
  void run(const uint8_t* chw, uint8_t* dst, size_t n_out) {
    static const char* in_names[] = {"image"};
    static const char* out_names[] = {"classes"};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t shape[4] = {1, 3, MODEL_H, MODEL_W};
    Ort::Value in = Ort::Value::CreateTensor<uint8_t>(mem, const_cast<uint8_t*>(chw), size_t(3) * MODEL_H * MODEL_W, shape, 4);
    Ort::RunOptions ro;
    auto out = session->Run(ro, in_names, &in, 1, out_names, 1);
    const auto info = out[0].GetTensorTypeAndShapeInfo();
    if (info.GetElementCount() != n_out) throw std::runtime_error("unexpected network output size");
    std::memcpy(dst, out[0].GetTensorData<uint8_t>(), n_out);
  }
};
}  // namespace

struct Engine::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "offroad_engine"};
  Net seg, contour, objseg;
  EngineConfig cfg;

  std::unique_ptr<Ort::Session> make(const std::string& path, std::string& provider) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    so.SetIntraOpNumThreads(1);                                   // the graph runs on the GPU: no CPU thread pool
    so.SetInterOpNumThreads(1);
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");    // no busy-waiting threads
    so.AddConfigEntry("session.inter_op.allow_spinning", "0");
    const OrtApi& api = Ort::GetApi();
    OrtCUDAProviderOptionsV2* cuda = nullptr;
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda));
    const std::string dev = std::to_string(cfg.device_id);
    const char* keys[] = {"device_id", "cudnn_conv_algo_search", "arena_extend_strategy"};
    const char* vals[] = {dev.c_str(), "HEURISTIC", "kSameAsRequested"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda, keys, vals, 3));
    OrtStatus* st = api.SessionOptionsAppendExecutionProvider_CUDA_V2(so, cuda);
    api.ReleaseCUDAProviderOptions(cuda);
    if (st) {
      const std::string msg = api.GetErrorMessage(st);
      api.ReleaseStatus(st);
      throw std::runtime_error("CUDA execution provider unavailable: " + msg);
    }
    provider = "CUDAExecutionProvider";
    return std::make_unique<Ort::Session>(env, path.c_str(), so);
  }
};

Engine::Engine(const EngineConfig& cfg) : impl_(new Impl) {
  impl_->cfg = cfg;
  const std::string d = cfg.onnx_dir + "/";
  if (cfg.seg) impl_->seg.session = impl_->make(d + "seg.onnx", provider_);
  if (cfg.contour) impl_->contour.session = impl_->make(d + "contour.onnx", provider_);
  if (cfg.objseg) impl_->objseg.session = impl_->make(d + "objseg.onnx", provider_);
  std::vector<uint8_t> dummy(size_t(3) * MODEL_H * MODEL_W, 127);
  EngineResult r;
  for (int i = 0; i < cfg.warmup; ++i) infer(dummy.data(), r);   // cuDNN / allocator warm-up
}

Engine::~Engine() = default;

void Engine::infer(const uint8_t* chw, EngineResult& out) {
  const auto t0 = Clock::now();
  const size_t n = size_t(MODEL_W) * MODEL_H;
  auto run_seg = [&] { const auto t = Clock::now(); out.seg.resize(n); impl_->seg.run(chw, out.seg.data(), n); out.ms_seg = ms_since(t); };
  auto run_con = [&] { const auto t = Clock::now(); impl_->contour.run(chw, out.contour, CONTOUR_RAYS); out.has_contour = true; out.ms_contour = ms_since(t); };
  auto run_obj = [&] { const auto t = Clock::now(); out.objseg.resize(n); impl_->objseg.run(chw, out.objseg.data(), n); out.ms_objseg = ms_since(t); };
  out.has_contour = false;
  if (impl_->cfg.parallel) {
    std::future<void> f1, f2;
    if (impl_->contour.session) f1 = std::async(std::launch::async, run_con);
    if (impl_->objseg.session) f2 = std::async(std::launch::async, run_obj);
    if (impl_->seg.session) run_seg();
    if (f1.valid()) f1.get();
    if (f2.valid()) f2.get();
  } else {
    if (impl_->seg.session) run_seg();
    if (impl_->contour.session) run_con();
    if (impl_->objseg.session) run_obj();
  }
  out.ms_total = ms_since(t0);
}

}  // namespace offroad
