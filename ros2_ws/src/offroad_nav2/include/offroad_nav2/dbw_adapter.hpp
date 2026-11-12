#pragma once
// Header-only port of offroad_vehicle_interface/dbw_adapter_base.py. No ROS dependency.
// TODO(user) decision 7.6: the real-car adapter (OSCC / ros2cco) is not chosen yet; only the logging stub exists.

#include <cstdarg>
#include <cstdio>
#include <functional>
#include <optional>
#include <string>
#include <utility>

namespace offroad {

struct DbwCommand {
  double steer_rad;
  double speed_mps;
  double accel;
};

struct DbwStatus {
  bool engaged = false;
  bool estop = false;
  std::optional<DbwCommand> last;
};

// Abstract drive-by-wire adapter.
class DbwAdapter {
 public:
  virtual ~DbwAdapter() = default;
  virtual void send(double steer_rad, double speed_mps, double accel) = 0;
  virtual DbwStatus status() const = 0;
  virtual void emergency_stop() = 0;
};

enum class LogLevel { Info, Warning };
using LogSink = std::function<void(LogLevel, const std::string&)>;

inline void stderr_log_sink(LogLevel level, const std::string& msg) {
  std::fprintf(stderr, "[%s] %s\n", level == LogLevel::Warning ? "WARN" : "INFO", msg.c_str());
}

inline std::string format_printf(const char* fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return buf;
}

// Stub: logs what it would send. Safe default; never touches hardware.
class LoggingDbwAdapter : public DbwAdapter {
 public:
  explicit LoggingDbwAdapter(LogSink sink = stderr_log_sink) : sink_(std::move(sink)) {}

  void send(double steer_rad, double speed_mps, double accel) override {
    if (estop_) {
      steer_rad = 0.0;
      speed_mps = 0.0;
      accel = 0.0;
    }
    last_ = DbwCommand{steer_rad, speed_mps, accel};
    sink_(LogLevel::Info, format_printf("DBW stub send steer=%.3f speed=%.2f accel=%.2f", steer_rad, speed_mps, accel));
  }

  DbwStatus status() const override {
    DbwStatus s;
    s.engaged = false;
    s.estop = estop_;
    s.last = last_;
    return s;
  }

  void emergency_stop() override {
    estop_ = true;
    last_ = DbwCommand{0.0, 0.0, 0.0};
    sink_(LogLevel::Warning, "DBW stub EMERGENCY STOP");
  }

  bool estop() const { return estop_; }

 private:
  LogSink sink_;
  std::optional<DbwCommand> last_;
  bool estop_ = false;
};

}  // namespace offroad
