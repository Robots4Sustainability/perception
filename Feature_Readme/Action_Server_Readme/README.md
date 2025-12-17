Terminal 1
```bash
ros2 launch perception perception_launch.py input_source:=robot
```

Terminal 2

```bash
cd src/perception/perception
python3 action_vision_manager.py
```

Terminal 3

```bash
ros2 action send_goal /run_vision_pipeline my_robot_interfaces/action/RunVision "{duration_seconds: 20.0}" --feedback
```

So it also implements a lifecycle such that it keeps the Yolo and Pose node on Standby. Only when the action is sent does it start to publish the frames for the duration specified.
