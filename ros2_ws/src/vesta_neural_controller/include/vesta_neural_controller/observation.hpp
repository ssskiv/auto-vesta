// Observation builder: a line-by-line port of vesta_rl/obs.py (the reference).
// Constants must match vesta_rl/config.py; test/test_observation.cpp checks
// this file against fixtures written by `ros2 run vesta_rl make_obs_fixture`.
#ifndef VESTA_NEURAL_CONTROLLER__OBSERVATION_HPP_
#define VESTA_NEURAL_CONTROLLER__OBSERVATION_HPP_

#include <cstddef>
#include <vector>

namespace vesta_neural_controller
{
namespace obs
{
constexpr double WHEELBASE = 2.648;
constexpr double MAX_STEER = 0.6;
constexpr double MAX_DV = 0.3;
constexpr double MAX_DDELTA = 0.1;

constexpr unsigned char LETHAL = 254;
constexpr unsigned char NO_INFORMATION = 255;

constexpr int PATCH_SIZE = 64;
constexpr double PATCH_RES = 0.25;
constexpr double PATCH_X_MIN = -4.0;
constexpr double PATCH_Y_MIN = -8.0;
constexpr float PATCH_UNKNOWN = 0.5f;

constexpr int PATH_POINTS = 20;
constexpr double PATH_SPACING = 1.0;
constexpr double PATH_SEARCH_DIST = 10.0;

constexpr double PATH_SCALE = 20.0;
constexpr double GOAL_SCALE = 50.0;
constexpr double YAW_RATE_SCALE = 1.0;
constexpr double CTE_SCALE = 5.0;

constexpr int VEC_SIZE = 2 * PATH_POINTS + 1 + 6;

// path speed limit (speedLimit); the speed action is a fraction of it
constexpr double A_LAT_MAX = 2.0;
constexpr double DECEL = 1.2;
constexpr double LIMIT_HORIZON = 12.0;
constexpr double V_GOAL = 1.0;
}  // namespace obs

double wrapAngle(double a);

// Nav2 costmap memory: data[my * size_x + mx], origin = corner of cell (0, 0).
struct CostmapView
{
  const unsigned char * data;
  int size_x;
  int size_y;
  double origin_x;
  double origin_y;
  double resolution;

  unsigned char costAt(double wx, double wy) const;
};

// out: PATCH_SIZE * PATCH_SIZE floats, row i = x forward, col j = y left.
void buildPatch(const CostmapView & costmap, double px, double py, double yaw, float * out);

struct Projection
{
  double x;
  double y;
  double s;
  double heading;
  std::size_t idx;
};

class PathTracker
{
public:
  PathTracker() = default;
  explicit PathTracker(const std::vector<double> & xy);  // x0, y0, x1, y1, ...

  // Replace the points (e.g. re-transformed into the costmap frame) keeping the
  // current index when the point count is unchanged.
  void setPoints(const std::vector<double> & xy);

  Projection peek(double px, double py) const;      // does not move idx
  Projection project(double px, double py);         // moves idx
  void pointAt(double s, double & x, double & y) const;

  double length() const {return s_.empty() ? 0.0 : s_.back();}
  std::size_t size() const {return x_.size();}
  bool empty() const {return x_.empty();}
  std::size_t idx = 0;

private:
  std::vector<double> x_, y_, seg_len_, s_;
};

struct VecInfo
{
  double s;
  double cte;
  double heading_err;
};

// out: VEC_SIZE floats. Calls tracker.project, so call once per control tick.
VecInfo buildVec(
  PathTracker & tracker, double px, double py, double yaw, double v, double yaw_rate,
  double last_v, double last_delta, double max_speed, float * out);

// Fastest speed at path position s from which every curve in the next
// LIMIT_HORIZON metres can be taken at <= A_LAT_MAX after braking at DECEL,
// and the end of the path reached at <= V_GOAL (vesta_rl obs.speed_limit).
double speedLimit(const PathTracker & tracker, double s, double max_speed);

// Policy action in [-1, 1]^2 -> (v, delta); a_v is a fraction of v_limit
// (speedLimit). Plus the per-tick rate limit.
void actionToCommand(float a_v, float a_delta, double v_limit, double & v, double & delta);
void rateLimit(double & v, double & delta, double last_v, double last_delta);

}  // namespace vesta_neural_controller

#endif  // VESTA_NEURAL_CONTROLLER__OBSERVATION_HPP_
