// Temporal filter on a 2D hit-count grid, BEV clustering and centroid tracking (pure C++, no ROS).
// Port of offroad_perception/tracking.py.
#pragma once
#include <Eigen/Dense>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace offroad {

using Vec2 = Eigen::Vector2d;

// Keep a point only if its cell (`cell` m, fixed world frame) was hit in >= min_hits of the last `window` frames (current included).
class TemporalFilter {
 public:
  explicit TemporalFilter(double cell = 0.2, int window = 3, int min_hits = 2) : cell_(cell), window_(window), min_hits_(min_hits) {}
  // xy: this frame's points (world frame). Returns a keep flag per point.
  std::vector<char> update(const std::vector<Vec2, Eigen::aligned_allocator<Vec2>>& xy);

 private:
  int64_t key(const Vec2& p) const;
  double cell_;
  int window_, min_hits_;
  std::deque<std::unordered_set<int64_t>> hist_;
};

struct Cluster {
  Vec2 center, size;
  int n = 0;
};
// Connected components (8-neighbourhood) of occupied BEV cells.
std::vector<Cluster> cluster_bev(const std::vector<Vec2, Eigen::aligned_allocator<Vec2>>& xy, double cell = 0.3, int min_pts = 3,
                                 int min_cells = 2);

// Nearest-centroid association with stable integer ids.
class CentroidTracker {
 public:
  explicit CentroidTracker(double gate = 1.5, int max_age = 5) : gate_(gate), max_age_(max_age) {}
  std::vector<std::pair<int, Cluster>> update(const std::vector<Cluster>& clusters);

 private:
  struct Track { Vec2 center; int age = 0; };
  double gate_;
  int max_age_, next_id_ = 1;
  std::unordered_map<int, Track> tracks_;
};

}  // namespace offroad
