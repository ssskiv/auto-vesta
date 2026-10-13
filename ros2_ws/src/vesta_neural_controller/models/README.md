Put the exported policy here as `policy.onnx` (`ros2 run vesta_rl export_onnx ...`),
then rebuild so it is installed to share/vesta_neural_controller/models/.
nav2_params.yaml points `FollowPath.model_path` at that installed file.
