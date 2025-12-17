# Project Contributions: Tool Detection & Pose Estimation

## 1. Project Restructuring
- **Standardization**: Refactored the codebase into a standard ROS 2 workspace structure (`ros2_ws/src/...`).
- **Dependencies**: Fixed `package.xml` and `setup.py` to ensure proper installation and visibility of the `perception` package.

## 2. Tool Detection (YOLOv8)
- **Training**: Trained a custom YOLOv8-small model on a 7-class tool dataset.
    - **Performance**: achieved **mAP@0.5 of 0.98** (High accuracy).
    - **Artifacts**: Model weights saved at `models/tools.pt`.
    - **Scripts**: Created `scripts/train_tool_model.py` for reproducible training.

## 3. Perception Pipeline
- **Integration**: Created `tool_pipeline.launch.py` to seamlessly launch:
    1.  RealSense Camera Driver (or Webcam).
    2.  YOLOv8 Node (Detection).
    3.  Pose Estimation Node (3D Analysis).
- **Topic Management**: Implemented clean topic remapping (`/tool_detector/detections`) to avoid conflicts with other subsystems.

## 4. Robust Pose Estimation (PCA)
Significantly improved the `pose_pca.py` algorithm:
- **Noise Reduction**: Implemented **Z-Score Outlier Removal** to ignore background noise/flying pixels.
- **Orientation Stability**:
    - **Z-Axis**: Forced normals to point *towards* the camera (preventing upside-down flips).
    - **X-Axis**: Implemented "Mass Distribution" check (skewness) to ensure the axis points towards the bulkier end of the tool (consistent direction).
- **Smoothing**: Added a **Low-Pass Temporal Filter** to reduce jitter variance in the published 6D pose.

## 5. Testing & Verification Tools
- **Mock Demo**: Created `demo_webcam_pose.launch.py` and `mock_depth_publisher.py` to verify the pipeline logic using a standard webcam (simulating depth data) when the RealSense is unavailable.
- **Hardware Diagnostics**: Created `scripts/test_realsense_windows.py` to quickly validate USB bandwidth and camera health.
