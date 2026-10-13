#include "vesta_neural_controller/observation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vesta_neural_controller
{

double wrapAngle(double a)
{
  // Python's (a + pi) % (2 pi) - pi: the result takes the sign of the divisor.
  const double two_pi = 2.0 * M_PI;
  double r = std::fmod(a + M_PI, two_pi);
  if (r < 0.0) {
    r += two_pi;
  }
  return r - M_PI;
}

unsigned char CostmapView::costAt(double wx, double wy) const
{
  const double fx = std::floor((wx - origin_x) / resolution);
  const double fy = std::floor((wy - origin_y) / resolution);
  if (fx < 0.0 || fy < 0.0 || fx >= size_x || fy >= size_y) {
    return obs::NO_INFORMATION;
  }
  return data[static_cast<int>(fy) * size_x + static_cast<int>(fx)];
}

void buildPatch(const CostmapView & costmap, double px, double py, double yaw, float * out)
{
  const double c = std::cos(yaw);
  const double s = std::sin(yaw);
  for (int i = 0; i < obs::PATCH_SIZE; ++i) {
    const double bx = obs::PATCH_X_MIN + (i + 0.5) * obs::PATCH_RES;
    for (int j = 0; j < obs::PATCH_SIZE; ++j) {
      const double by = obs::PATCH_Y_MIN + (j + 0.5) * obs::PATCH_RES;
      const double wx = px + c * bx - s * by;
      const double wy = py + s * bx + c * by;
      const unsigned char cost = costmap.costAt(wx, wy);
      out[i * obs::PATCH_SIZE + j] = cost == obs::NO_INFORMATION ?
        obs::PATCH_UNKNOWN : static_cast<float>(cost) / static_cast<float>(obs::LETHAL);
    }
  }
}

PathTracker::PathTracker(const std::vector<double> & xy)
{
  setPoints(xy);
  idx = 0;
}

void PathTracker::setPoints(const std::vector<double> & xy)
{
  const std::size_t n = xy.size() / 2;
  if (n != x_.size()) {
    idx = 0;
  }
  x_.resize(n);
  y_.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    x_[i] = xy[2 * i];
    y_[i] = xy[2 * i + 1];
  }
  seg_len_.assign(n > 0 ? n - 1 : 0, 0.0);
  s_.assign(n, 0.0);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    seg_len_[i] = std::hypot(x_[i + 1] - x_[i], y_[i + 1] - y_[i]);
    s_[i + 1] = s_[i] + seg_len_[i];
  }
}

Projection PathTracker::peek(double px, double py) const
{
  const std::size_t n = x_.size();
  if (n == 1) {
    return {x_[0], y_[0], 0.0, 0.0, 0};
  }
  std::size_t best_i = idx;
  double best_t = 0.0;
  double best_d2 = std::numeric_limits<double>::infinity();
  const double limit = s_[idx] + obs::PATH_SEARCH_DIST;
  for (std::size_t i = idx; i < n - 1 && s_[i] <= limit; ++i) {
    const double ax = x_[i], ay = y_[i];
    const double dx = x_[i + 1] - ax, dy = y_[i + 1] - ay;
    const double l2 = dx * dx + dy * dy;
    const double t = l2 <= 0.0 ? 0.0 :
      std::min(1.0, std::max(0.0, ((px - ax) * dx + (py - ay) * dy) / l2));
    const double qx = ax + t * dx, qy = ay + t * dy;
    const double d2 = (px - qx) * (px - qx) + (py - qy) * (py - qy);
    if (d2 < best_d2) {
      best_i = i;
      best_t = t;
      best_d2 = d2;
    }
  }
  const double ax = x_[best_i], ay = y_[best_i];
  const double bx = x_[best_i + 1], by = y_[best_i + 1];
  Projection p;
  p.x = ax + best_t * (bx - ax);
  p.y = ay + best_t * (by - ay);
  p.s = s_[best_i] + best_t * seg_len_[best_i];
  p.heading = std::atan2(by - ay, bx - ax);
  p.idx = best_i;
  return p;
}

Projection PathTracker::project(double px, double py)
{
  const Projection p = peek(px, py);
  idx = p.idx;
  return p;
}

void PathTracker::pointAt(double s, double & x, double & y) const
{
  const std::size_t n = x_.size();
  if (n == 1 || s >= length()) {
    x = x_[n - 1];
    y = y_[n - 1];
    return;
  }
  if (s <= 0.0) {
    x = x_[0];
    y = y_[0];
    return;
  }
  // numpy.searchsorted(s_, s, side='right') - 1
  long j = static_cast<long>(std::upper_bound(s_.begin(), s_.end(), s) - s_.begin()) - 1;
  j = std::min(std::max(j, 0L), static_cast<long>(n) - 2);
  const double t = seg_len_[j] > 0.0 ? (s - s_[j]) / seg_len_[j] : 0.0;
  x = x_[j] + t * (x_[j + 1] - x_[j]);
  y = y_[j] + t * (y_[j + 1] - y_[j]);
}

namespace
{
void toBase(double px, double py, double yaw, double x, double y, double & bx, double & by)
{
  const double dx = x - px, dy = y - py;
  const double c = std::cos(yaw), s = std::sin(yaw);
  bx = c * dx + s * dy;
  by = -s * dx + c * dy;
}

double clamp(double v, double lo, double hi) {return std::min(hi, std::max(lo, v));}
}  // namespace

VecInfo buildVec(
  PathTracker & tracker, double px, double py, double yaw, double v, double yaw_rate,
  double last_v, double last_delta, double max_speed, float * out)
{
  const Projection p = tracker.project(px, py);
  for (int k = 0; k < obs::PATH_POINTS; ++k) {
    double x, y, bx, by;
    tracker.pointAt(p.s + (k + 1) * obs::PATH_SPACING, x, y);
    toBase(px, py, yaw, x, y, bx, by);
    out[2 * k] = static_cast<float>(bx / obs::PATH_SCALE);
    out[2 * k + 1] = static_cast<float>(by / obs::PATH_SCALE);
  }
  const int o = 2 * obs::PATH_POINTS;
  out[o] = static_cast<float>(clamp((tracker.length() - p.s) / obs::GOAL_SCALE, 0.0, 1.0));
  double unused, cte;
  toBase(px, py, yaw, p.x, p.y, unused, cte);
  const double heading_err = wrapAngle(p.heading - yaw);
  out[o + 1] = static_cast<float>(v / max_speed);
  out[o + 2] = static_cast<float>(yaw_rate / obs::YAW_RATE_SCALE);
  out[o + 3] = static_cast<float>(last_v / max_speed);
  out[o + 4] = static_cast<float>(last_delta / obs::MAX_STEER);
  out[o + 5] = static_cast<float>(clamp(cte / obs::CTE_SCALE, -1.0, 1.0));
  out[o + 6] = static_cast<float>(heading_err / M_PI);
  return {p.s, cte, heading_err};
}

double speedLimit(const PathTracker & tracker, double s, double max_speed)
{
  const double step = 1.0;
  const double to_end = std::max(0.0, tracker.length() - s);
  double limit = std::min(max_speed, std::sqrt(obs::V_GOAL * obs::V_GOAL + 2.0 * obs::DECEL * to_end));
  const int n = static_cast<int>(obs::LIMIT_HORIZON / step) + 3;
  // from 1 step behind the car: the heading change at pts[k] is (k - 1) * step ahead
  std::vector<double> px(n), py(n);
  for (int k = 0; k < n; ++k) {
    tracker.pointAt(s + (k - 1) * step, px[k], py[k]);
  }
  bool have_prev = false;
  double prev_h = 0.0;
  for (int k = 0; k < n - 1; ++k) {
    const double dx = px[k + 1] - px[k], dy = py[k + 1] - py[k];
    if (std::abs(dx) + std::abs(dy) < 1e-6) {  // past the end of the path
      break;
    }
    const double h = std::atan2(dy, dx);
    if (have_prev) {
      const double kappa = std::abs(wrapAngle(h - prev_h)) / step;
      const double v_curve = kappa > 1e-6 ? std::sqrt(obs::A_LAT_MAX / kappa) : max_speed;
      limit = std::min(limit, std::sqrt(v_curve * v_curve + 2.0 * obs::DECEL * (k - 1) * step));
    }
    prev_h = h;
    have_prev = true;
  }
  return limit;
}

void actionToCommand(float a_v, float a_delta, double v_limit, double & v, double & delta)
{
  const double av = clamp(a_v, -1.0, 1.0);
  const double ad = clamp(a_delta, -1.0, 1.0);
  v = (av + 1.0) * 0.5 * v_limit;
  delta = ad * obs::MAX_STEER;
}

void rateLimit(double & v, double & delta, double last_v, double last_delta)
{
  v = clamp(v, last_v - obs::MAX_DV, last_v + obs::MAX_DV);
  delta = clamp(delta, last_delta - obs::MAX_DDELTA, last_delta + obs::MAX_DDELTA);
}

}  // namespace vesta_neural_controller
