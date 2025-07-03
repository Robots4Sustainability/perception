# Vision-Based Object Detection and Pose Estimation Framework

This repository provides a complete vision pipeline that supports object detection with YOLOv8 and 6D pose estimation. The project is structured to support fine-tuning, inference, and integration with robotic systems using ROS 2.

**Python version:** 3.12  
**ROS 2:** Jazzy

---

## Table of Contents

- [Cloning the Repository](#cloning-the-repository)
- [Fine-Tuning Instructions](#fine-tuning-instructions)
- [Pose Estimation Setup](#pose-estimation-setup)
- [Project Structure](#project-structure)
- [Build Instructions](#build-instructions)
- [Launching the Camera and Robot](#launching-the-camera-and-robot)
- [Running the Nodes](#running-the-nodes)
- [RViz Visualization](#rviz-visualization)
- [Topics](#topics)

---

## Cloning the Repository

```bash
git clone git@github.com:Robots4Sustainability/perception.git
cd your-repo
```

---

## Fine-Tuning Instructions

1. Navigate to the fine-tuning folder:
   ```bash
   cd Annotation_and_fine-tuning
   ```

2. Create a Python virtual environment and activate it:
   ```bash
   python3.12 -m venv venvTrain
   source venvTrain/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from the [Releases](https://github.com/Robots4Sustainability/perception/releases/tag/v1.0.0) page.

5. Run `main.ipynb`:
   - Renames files (if needed)
   - Applies a Vision Transformer to assist in image labeling
   - Displays YOLOv8-compatible bounding boxes
   - Trains a YOLOv8 model

   The trained model will be saved at:
   ```
   runs/detect/xxxx/weights/best.pt
   ```

6. Use `Inference.ipynb` to test the trained model.

---

## Pose Estimation Setup

1. Create a virtual environment in the root folder:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Unzip the `master_project.zip`:
   ```bash
   unzip master_project.zip
   cd master_project
   rm -rf build log install
   cd ..
   ```

3. Clean the `ros2_ws` workspace:
   ```bash
   cd ros2_ws
   rm -rf build log install
   cd ..
   ```

---

## Project Structure

This project contains two workspaces:

- **`master_project`**: Contains nodes specific to the Kinova robot and accesses the `ros2_kortex_vision` package.
- **`ros2_ws`**: Contains the main logic for YOLOv8 detection and 6D pose estimation.

---

## Build Instructions

### Build `master_project`
```bash
cd master_project
colcon build
source install/setup.bash
```

### Build `ros2_ws`
```bash
cd ros2_ws
colcon build --packages-select perception
source install/setup.bash
```

---

## Launching the Camera and Robot

### RealSense Camera Launch
```bash
ros2 launch realsense2_camera rs_launch.py \
  enable_rgbd:=true \
  enable_sync:=true \
  align_depth.enable:=true \
  enable_color:=true \
  enable_depth:=true \
  pointcloud.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  pointcloud.ordered_pc:=true
```

### Kinova Robot Vision Node Launch
Left Arm IP: 192.168.1.10
Right Arm IP: 192.168.1.12

```bash
ros2 launch kinova_vision kinova_vision.launch.py \
  device:=192.168.1.12 \
  depth_registration:=true \
  color_camera_info_url:=package://kinova_vision/launch/calibration/default_color_calib_1280x720.ini
```

---

## Running the Nodes

> **Important**: Before running, ensure you update the absolute path in `src/perception/perception/yolo_object_detection.py`.

### Run YOLO Object Detection

```bash
ros2 run perception yolo_node --ros-args \
  -p input_mode:=default \
  -p model_type:=fine_tuned
```

- `input_mode`: `robot` or `realsense` (default) 
- `model_type`: `fine_tuned` or `default` (YOLOv8n)

### Run Pose Estimation

```bash
ros2 run perception pose_node --ros-args -p input_mode:=default
```

- `input_mode`: `robot` or `realsense` (default) 

---

## RViz Visualization

To visualize the results:

```bash
rviz2
```

You can load the pre-configured RViz setup:

```bash
ros2_ws/src/perception/rviz/pose_estimation.rviz
```

---

## Topics

| Topic Name             | Description                         |
|------------------------|-------------------------------------|
| `/annotated_images`    | Publishes YOLO-annotated image data |
| `/cropped_pointcloud`  | Publishes cropped point cloud used for pose estimation |
