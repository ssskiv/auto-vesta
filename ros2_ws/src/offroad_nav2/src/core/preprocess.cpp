#include "offroad_nav2/preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include "offroad_nav2/geometry.hpp"

namespace offroad {

void preprocess_rgb_chw(const uint8_t* img, int width, int height, int step, int channels, bool bgr, bool crop, int crop_top,
                        int crop_bottom, uint8_t* out) {
  const int y0 = crop ? crop_top : 0, y1 = crop ? height - crop_bottom : height;
  cv::Mat src(y1 - y0, width, channels == 4 ? CV_8UC4 : CV_8UC3, const_cast<uint8_t*>(img) + size_t(y0) * step, step);
  cv::Mat small;
  if (src.cols == MODEL_W && src.rows == MODEL_H) small = src;
  else cv::resize(src, small, cv::Size(MODEL_W, MODEL_H), 0, 0, cv::INTER_LINEAR);
  const int c0 = bgr ? 2 : 0, c2 = bgr ? 0 : 2;   // source channel feeding the R and the B plane
  const size_t plane = size_t(MODEL_W) * MODEL_H;
  uint8_t *pr = out, *pg = out + plane, *pb = out + 2 * plane;
  for (int y = 0; y < MODEL_H; ++y) {
    const uint8_t* row = small.ptr<uint8_t>(y);
    const size_t o = size_t(y) * MODEL_W;
    for (int x = 0; x < MODEL_W; ++x) {
      const uint8_t* px = row + size_t(x) * channels;
      pr[o + x] = px[c0];
      pg[o + x] = px[1];
      pb[o + x] = px[c2];
    }
  }
}

}  // namespace offroad
