### Run Table Segmentation

Applies segmentation on the table and detects empty spaces.

```bash
ros2 run perception pose_node --ros-args -p input_mode:=default
```

**Publish object radius:**

In another terminal, run:
```bash
ros2 topic pub --once /perception/detected_object_radius std_msgs/msg/Float32 "{data: <object_radius>}"
```

For example, <object_radius> could be 0.07.
