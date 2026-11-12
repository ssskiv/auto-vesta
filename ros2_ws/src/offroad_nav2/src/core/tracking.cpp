#include "offroad_nav2/tracking.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

namespace offroad {

int64_t TemporalFilter::key(const Vec2& p) const {
  const int64_t ix = int64_t(std::floor(p.x() / cell_)), iy = int64_t(std::floor(p.y() / cell_));
  return (ix << 32) ^ (iy & 0xffffffffLL);
}

std::vector<char> TemporalFilter::update(const std::vector<Vec2, Eigen::aligned_allocator<Vec2>>& xy) {
  std::vector<char> keep(xy.size(), 0);
  if (xy.empty()) {
    hist_.emplace_back();
    while (int(hist_.size()) > window_) hist_.pop_front();
    return keep;
  }
  std::vector<int64_t> keys(xy.size());
  std::unordered_set<int64_t> cur;
  for (size_t i = 0; i < xy.size(); ++i) { keys[i] = key(xy[i]); cur.insert(keys[i]); }
  std::unordered_map<int64_t, int> counts;
  for (auto k : cur) {
    int c = 1;
    for (const auto& s : hist_) c += s.count(k) ? 1 : 0;
    counts[k] = c;
  }
  hist_.push_back(std::move(cur));
  while (int(hist_.size()) > window_) hist_.pop_front();
  for (size_t i = 0; i < xy.size(); ++i) keep[i] = counts[keys[i]] >= min_hits_;
  return keep;
}

std::vector<Cluster> cluster_bev(const std::vector<Vec2, Eigen::aligned_allocator<Vec2>>& xy, double cell, int min_pts, int min_cells) {
  std::vector<Cluster> out;
  if (xy.empty()) return out;
  std::vector<int64_t> kx(xy.size()), ky(xy.size());
  int64_t minx = INT64_MAX, miny = INT64_MAX, maxx = INT64_MIN, maxy = INT64_MIN;
  for (size_t i = 0; i < xy.size(); ++i) {
    kx[i] = int64_t(std::floor(xy[i].x() / cell));
    ky[i] = int64_t(std::floor(xy[i].y() / cell));
    minx = std::min(minx, kx[i]); miny = std::min(miny, ky[i]); maxx = std::max(maxx, kx[i]); maxy = std::max(maxy, ky[i]);
  }
  const int64_t W = maxx - minx + 3, H = maxy - miny + 3;
  if (W * H > 4000000) return out;
  cv::Mat grid = cv::Mat::zeros(int(W), int(H), CV_8U);
  for (size_t i = 0; i < xy.size(); ++i) grid.at<uint8_t>(int(kx[i] - minx + 1), int(ky[i] - miny + 1)) = 1;
  cv::Mat lab, stats, cen;
  const int n = cv::connectedComponentsWithStats(grid, lab, stats, cen, 8, CV_32S);
  std::vector<Vec2, Eigen::aligned_allocator<Vec2>> lo(n, Vec2::Constant(1e18)), hi(n, Vec2::Constant(-1e18)), sum(n, Vec2::Zero());
  std::vector<int> cnt(n, 0);
  for (size_t i = 0; i < xy.size(); ++i) {
    const int l = lab.at<int>(int(kx[i] - minx + 1), int(ky[i] - miny + 1));
    ++cnt[l];
    sum[l] += xy[i];
    lo[l] = lo[l].cwiseMin(xy[i]);
    hi[l] = hi[l].cwiseMax(xy[i]);
  }
  for (int i = 1; i < n; ++i) {
    if (cnt[i] < min_pts || stats.at<int>(i, cv::CC_STAT_AREA) < min_cells) continue;
    Cluster c;
    c.center = sum[i] / cnt[i];
    c.size = (hi[i] - lo[i]).cwiseMax(cell);
    c.n = cnt[i];
    out.push_back(c);
  }
  return out;
}

std::vector<std::pair<int, Cluster>> CentroidTracker::update(const std::vector<Cluster>& clusters) {
  std::unordered_set<int> used;
  std::vector<std::pair<int, Cluster>> res;
  for (const auto& c : clusters) {
    int best = -1;
    double bd = gate_;
    for (const auto& [tid, t] : tracks_) {
      const double d = (t.center - c.center).norm();
      if (!used.count(tid) && d < bd) { best = tid; bd = d; }
    }
    if (best < 0) best = next_id_++;
    used.insert(best);
    tracks_[best] = Track{c.center, 0};
    res.emplace_back(best, c);
  }
  for (auto it = tracks_.begin(); it != tracks_.end();) {
    if (!used.count(it->first) && ++it->second.age > max_age_) it = tracks_.erase(it);
    else ++it;
  }
  return res;
}

}  // namespace offroad
