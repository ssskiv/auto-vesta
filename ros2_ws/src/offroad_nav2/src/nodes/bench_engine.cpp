// Engine benchmark: p50 / p95 of preprocessing + the three networks (+ class maps back on the CPU) for a 1280x720 bgra frame.
// Same scope as the Python tool `offroad_perception.bench_engine`. Usage: bench_engine [--n 300] [--parallel] [--onnx-dir DIR]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "offroad_nav2/engine.hpp"
#include "offroad_nav2/preprocess.hpp"

using Clock = std::chrono::steady_clock;
static double pct(std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, size_t(q * v.size()))]; }

int main(int argc, char** argv) {
  int n = 300;
  offroad::EngineConfig cfg;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--n") && i + 1 < argc) n = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--parallel")) cfg.parallel = true;
    else if (!std::strcmp(argv[i], "--onnx-dir") && i + 1 < argc) cfg.onnx_dir = argv[++i];
  }
  offroad::Engine eng(cfg);
  std::vector<uint8_t> img(size_t(1280) * 720 * 4), chw(size_t(3) * 320 * 640);
  std::mt19937 rng(0);
  for (auto& b : img) b = uint8_t(rng());
  offroad::EngineResult r;
  for (int i = 0; i < 20; ++i) { offroad::preprocess_rgb_chw(img.data(), 1280, 720, 1280 * 4, 4, true, false, 0, 0, chw.data()); eng.infer(chw.data(), r); }
  std::vector<double> tot, pre, seg, con, obj;
  for (int i = 0; i < n; ++i) {
    const auto t0 = Clock::now();
    offroad::preprocess_rgb_chw(img.data(), 1280, 720, 1280 * 4, 4, true, false, 0, 0, chw.data());
    const auto t1 = Clock::now();
    eng.infer(chw.data(), r);
    const auto t2 = Clock::now();
    tot.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());
    pre.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    seg.push_back(r.ms_seg); con.push_back(r.ms_contour); obj.push_back(r.ms_objseg);
  }
  std::printf("p50 / p95 ms over %d frames (1280x720 input, fp16, provider %s, networks %s)\n", n, eng.provider().c_str(),
              cfg.parallel ? "in parallel threads" : "sequential");
  std::printf("preprocess+3 nets+class maps:  %6.1f / %6.1f\n", pct(tot, .5), pct(tot, .95));
  std::printf("  preprocess %6.1f / %6.1f\n  seg        %6.1f / %6.1f\n  contour    %6.1f / %6.1f\n  objseg     %6.1f / %6.1f\n", pct(pre, .5),
              pct(pre, .95), pct(seg, .5), pct(seg, .95), pct(con, .5), pct(con, .95), pct(obj, .5), pct(obj, .95));
  return 0;
}
