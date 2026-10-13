// ONNX Runtime wrapper for the policy exported by `ros2 run vesta_rl export_onnx`.
// Built without onnxruntime, load() always fails and the controller runs on the
// fallback only.
#ifndef VESTA_NEURAL_CONTROLLER__ONNX_POLICY_HPP_
#define VESTA_NEURAL_CONTROLLER__ONNX_POLICY_HPP_

#include <memory>
#include <string>

namespace vesta_neural_controller
{

class OnnxPolicy
{
public:
  OnnxPolicy();
  ~OnnxPolicy();

  // Checks that the model takes costmap [N, 1, PATCH_SIZE, PATCH_SIZE] and vec [N, VEC_SIZE].
  bool load(const std::string & path, std::string & error);
  bool loaded() const;
  // costmap: PATCH_SIZE^2 floats, vec: VEC_SIZE floats, action: 2 floats in [-1, 1].
  bool infer(const float * costmap, const float * vec, float * action, std::string & error);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vesta_neural_controller

#endif  // VESTA_NEURAL_CONTROLLER__ONNX_POLICY_HPP_
