# Table Height Estimation

Estimates the height of a table surface from a depth camera point cloud.
Operates in the `eddie_base_footprint` world frame (Z = 0 = floor) — no floor
detection needed. Uses iterative RANSAC to strip walls/non-horizontal planes
until a horizontal surface is found.

**Published topics**

| Topic | Type | Description |
|---|---|---|
| `/table_height_value` | `Float32` | Table height in metres |
| `/table_height_visualization` | `MarkerArray` | RViz markers (sphere, line, label) |
| `/table_points_debug` | `PointCloud2` | RANSAC inlier points (green cloud) |

---

## Running on the live robot

Open two terminals.

**Terminal 1 — start and activate the node**
```bash
source /opt/ros/jazzy/setup.bash
source ~/r4s-ws/install/setup.bash

ros2 run perception asjad_node --ros-args -p input_mode:=robot
```

In a second terminal, activate the lifecycle node:
```bash
source /opt/ros/jazzy/setup.bash
ros2 lifecycle set /table_height_estimator configure
ros2 lifecycle set /table_height_estimator activate
```

**Terminal 2 — open RViz with the pre-built config**
```bash
source /opt/ros/jazzy/setup.bash
rviz2 -d ~/r4s-ws/src/perception/config/table_height.rviz
```

Point the camera at a table. The terminal prints `Table height: X.XXX m (world frame)`
and RViz shows the annotated point cloud with a height label.

---

## Testing with ROS bags (no robot needed)

### 1. Clone the repo and check out the branch

```bash
git clone <repo-url> ~/Desktop/table_height_ros
cd ~/Desktop/table_height_ros/perception
git checkout table-height-eddie-base
```

### 2. Install dependencies

```bash
pip install open3d scipy --break-system-packages
```

### 3. Get the bag files

Download the two bag zip files from the shared Google Drive folder and place them
inside `bags/for_asjad/`, then extract:

```bash
cd ~/Desktop/table_height_ros/bags/for_asjad
unzip table_height_with_robot.zip  -d bag1/
unzip table_height_with_robot2.zip -d bag2/
```

After extraction the layout should be:
```
bags/for_asjad/bag1/table_height_with_robot/    ← bag 1
bags/for_asjad/bag2/table_height_with_robot2/   ← bag 2
```

### 4. Run the demo

The helper script at the repo root starts the node, activates it, opens RViz
with the pre-built config (`config/table_height.rviz`), and plays the bag on loop.

```bash
cd ~/Desktop/table_height_ros

# Bag 1 (default) — table at ~0.70 m, robot-mounted camera
bash run_rviz_demo.sh

# Bag 2 — table at ~0.72 m, elevated camera angle
bash run_rviz_demo.sh 2
```

Press `Ctrl+C` to stop everything.

### 5. Expected results

| | Bag 1 | Bag 2 |
|---|---|---|
| Reported height | ~0.700 m | ~0.720 m |
| RViz | Green table cloud, red sphere, white label | Same |
| Terminal | `Table height: 0.700 m (world frame)` | `Table height: 0.720 m (world frame)` |
