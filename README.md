# Vision-Based Object Detection and Pose Estimation Framework

This repository provides a complete vision pipeline that supports object detection with YOLOv8 and 6D pose estimation. The project is structured to support fine-tuning, inference, and integration with robotic systems using ROS 2.

**Python version:** 3.12  
**ROS 2:** Jazzy

---

## Table of Contents

- [Release_3](#Release_3)
- [Cloning the Repository](#cloning-the-repository)
- [Project Structure](#project-structure)
- [Pose Estimation Setup](#pose-estimation-setup)
- [Launching the Camera and Robot](#launching-the-camera-and-robot)
- [Running the Nodes](#running-the-nodes)
  - [Option 1: All-in-One Launch (Camera + Nodes)](#1-both-camera-and-the-nodes)
  - [Option 2: Nodes-Only Launch](#2-all-the-nodes-via-one-launch-file)
  - [Option 3: Separate Nodes/Terminals](#3-run-yolo-object-detection)
- [RViz Visualization](#rviz-visualization)
- [Topics](#topics)
- [Fine-Tuning Instructions](#fine-tuning-instructions)

---

## Cloning the Repository

```bash
git clone git@github.com:Robots4Sustainability/perception.git
cd your-repo
```

---


## Project Structure

- **`ros_perception_scripts`**: Contains nodes specific scripts which goes inside the src folder.
- **`Annotation_and_fine-tuning`**: Contains scripts/utility scripts if you wish to train/fine-tune on your custom dataset.

---

## Pose Estimation Setup

1. Create a virtual environment in the root folder:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies (Use the **root folder's** requirements.txt):
   ```bash
   pip install -r requirements.txt
   ```

3. Follow [documentation](https://github.com/Robots4Sustainability/documentation) on how to set up the robot and ROS2 workspace. 
   
4. Once eddie_ros, kinova_vision and ROS2 workspace has been setup. Install this package
   ```bash
   sudo apt update
   sudo apt install ros-jazzy-image-pipeline
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

For Kinova Vision Node, you must `export RMW_IMPLEMENTATION=rmw_zenoh_cpp` in every terminal, but before that
Follow the instructions below:
```bash
sudo apt install ros-jazzy-rmw-zenoh-cpp

pkill -9 -f ros && ros2 daemon stop
```
[Install Zenoh router](https://zenoh.io/docs/getting-started/installation/) and then `source /opt/ros/jazzy/setup.bash`

```bash
# terminal 1
ros2 run rmw_zenoh_cpp rmw_zenohd


# terminal 2
export RMW_IMPLEMENTATION=rmw_zenoh_cpp

your kinova vision node
```

Command to run kinova_vision.lanch

```bash
ros2 launch kinova_vision kinova_vision.launch.py \
  device:=192.168.1.12 \
  depth_registration:=true \
  color_camera_info_url:=package://kinova_vision/launch/calibration/default_color_calib_1280x720.ini
```

If you get resolution errors then, go to the admin portal of the arm(192.168.1.1X) and then in the Camera settings, set the calibration to 1280x720

---

## Running the Nodes

The project provides multiple ways to launch the files.
1. Both Camera and the Nodes
2. All the Nodes via one launch file
3. Separate Nodes/Terminals

### 1. Both Camera and the Nodes

```bash
ros2 launch perception camera_and_perception_launch.py input_source:=robot
```

- `input_source`: `robot` or `realsense` (default)

### 2. All the Nodes via one launch file

In this, you must run the camera in a separate terminal and rest of the nodes will be executed under one launch file

```bash
ros2 launch perception perception_launch.py input_source:=robot
```

- `input_source`: `robot` or `realsense` (default)

### 3. Run YOLO Object Detection

```bash
ros2 run perception yolo_node --ros-args \
  -p input_mode:=realsense \
  -p model_type:=fine_tuned \
  -p conf_threshold:=0.6 \
  -p device:="cpu"
```

- `input_mode`: `robot` or `realsense` (default)
- `model_type`: `fine_tuned` or `default` (YOLOv8n COCO Dataset)
- `model_path`: optional explicit path to weights (.pt). If empty, falls back to defaults based on `model_type`.
- `conf_threshold`: float, default `0.6`.
- `device`: optional device string (e.g., `cpu`, `cuda`).
- `class_names`: optional list to override class names; must align with training order.

### 3. Run Pose Estimation

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
| `/pickable_objects`    | Publishes YOLO-detection array for objects which can be picked by robot (see `object_classifier_node.py`) |
| `/non_pickable_objects`| Publishes YOLO-detection array for objects which can't be picked by robot (see `object_classifier_node.py`) |

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

4. Download the dataset from the [Releases](https://github.com/Robots4Sustainability/perception/releases) page.

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

# Release_3
[Table_Segmentation](https://github.com/Robots4Sustainability/perception/tree/dev/Feature_Readme/Table_Segmentation_Readme)
[Table_Height](https://github.com/Robots4Sustainability/perception/blob/dev/Feature_Readme/Table_Height_Readme/README.md)
[Action_Server](https://github.com/Robots4Sustainability/perception/blob/dev/Feature_Readme/Action_Server_Readme/README.md)
[Tool_Detection](https://github.com/Robots4Sustainability/perception/blob/dev/Feature_Readme/Tool_Detection/README.md)
