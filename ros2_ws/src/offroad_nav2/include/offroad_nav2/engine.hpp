// Inference engine: three separate networks (FreespaceSeg, FreespaceContour, ObjectSeg) on one preprocessed frame, ONNX Runtime
// with the CUDA provider. Built once, fp16 (the fp16 conversion, ImageNet normalisation and argmax live inside the ONNX graphs,
// see tools/export_onnx.py), so only a tiny uint8 class map comes back from the GPU.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "offroad_nav2/geometry.hpp"

namespace offroad {

struct EngineConfig {
  std::string onnx_dir = "/models/onnx";   // seg.onnx, contour.onnx, objseg.onnx
  bool seg = true, contour = true, objseg = true;
  int device_id = 0;
  bool parallel = false;                   // run the three networks from three threads (separate CUDA streams)
  int warmup = 5;
};

struct EngineResult {
  std::vector<uint8_t> seg, objseg;        // 320 x 640 class maps (empty when the network is disabled)
  uint8_t contour[CONTOUR_RAYS] = {0};
  bool has_contour = false;
  double ms_seg = 0, ms_contour = 0, ms_objseg = 0, ms_total = 0;
};

class Engine {
 public:
  explicit Engine(const EngineConfig& cfg);
  ~Engine();
  // chw: 3 x 320 x 640 uint8 RGB planar (see preprocess_rgb_chw)
  void infer(const uint8_t* chw, EngineResult& out);
  const std::string& provider() const { return provider_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string provider_;
};

}  // namespace offroad
