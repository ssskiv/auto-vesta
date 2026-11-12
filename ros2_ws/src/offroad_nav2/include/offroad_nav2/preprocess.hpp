// Image -> 640x320 uint8 RGB planar (CHW) tensor for the networks. Normalisation, fp16 and argmax are inside the ONNX graphs.
#pragma once
#include <cstdint>

namespace offroad {

// img: interleaved, `channels` = 3 or 4 (bgr/bgra when `bgr`, rgb/rgba otherwise), row stride `step` bytes.
// mode 'resize' squashes the full frame (matches training); 'crop' drops crop_top/crop_bottom rows first.
// out: 3 * 320 * 640 bytes.
void preprocess_rgb_chw(const uint8_t* img, int width, int height, int step, int channels, bool bgr, bool crop, int crop_top,
                        int crop_bottom, uint8_t* out);

}  // namespace offroad
