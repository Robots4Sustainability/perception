# 🤖 Perception Pipeline — Final Release
 
> **Robots4Sustainability** | ROS 2 Vision Pipeline for Robotic Object Detection & Pose Estimation  
> Built on YOLOv8 · Open3D · Kinova Arm Platform · ROS 2 Jazzy
 
---
 
## 📋 Table of Contents
 
- [Setup & Configuration](#setup--configuration)
- [Running the Perception Pipeline](#running-the-perception-pipeline)
- [Further Reading](#-further-reading)
 
---
 
## Setup & Configuration
 
For full prerequisites, environment setup, cloning, virtual environment creation, ROS 2 workspace build, and Zenoh middleware configuration:
 
👉 [Setup & Installation Wiki](https://github.com/Robots4Sustainability/perception/wiki/Setup-and-Installation)
 
---
 
## Running the Perception Pipeline
 
Launch the pipeline across **three terminals** in order.
 
---
 
Terminal 1 — Camera
 
Launch the Kinova Vision camera node:
 
```bash
ros2 launch kinova_vision kinova_vision.launch.py \
    device:=192.168.1.12 \
    depth_registration:=true \
    color_camera_info_url:=file:///home/r4s/r4s-ws/src/ros2_kortex_vision/launch/calibration/default_color_calib_1280x720.ini \
    depth_camera_info_url:=file:///home/r4s/r4s-ws/src/ros2_kortex_vision/launch/calibration/default_depth_calib_480x270.ini
```
 
📖 [Camera Configuration Wiki](https://github.com/Robots4Sustainability/perception/wiki/Camera-Configuration)

 
Terminal 2 — Action Dispatcher
 
> ⚠️ **Remember to activate your virtual environment before running.**
 
Launch all perception nodes via the action server dispatcher:
 
```bash
ros2 launch perception action_launch.py lazy_loading:=true
```

### Terminal 3 - Brain Client (Optional)
 
**If the FSM is not running**, you can alternatively run the brain client directly or send commands manually:
 
```bash
# Option A — Run brain client directly
python brain_client.py car_objects motor_grip 0.0
 
# Option B — Send a direct task command
ros2 action send_goal /run_perception_pipeline   my_robot_interfaces/action/RunVision   "{task_name: 'car_objects', object_class: 'motor_grip'}"   --feedback
```
📖 [Action Server & Brain Client Wiki](https://github.com/Robots4Sustainability/perception/wiki/Action-Server-&-Brain-Client)
 
---
 
## 📚 Further Reading
[Full Documentation](https://github.com/Robots4Sustainability/perception/wiki)
