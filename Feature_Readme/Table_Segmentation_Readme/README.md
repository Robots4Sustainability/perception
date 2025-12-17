### Run Object Segmentation
```bash
ros2 run perception segment_object
```

### Run Table Segmentation

Applies segmentation on the table and detects empty spaces.

```bash
ros2 run perception table_segmentation_node --ros-args -p mode:=test -p test_radius:=0.08
```
- `input_mode`: `robot`(default) or `realsense` 
- `mode`: `test` or `actual` (`actual`=listening to publisher msg, `test`=random test radius values default 0.07 or 7cm) 
- `safety_margin`: float, default `0.02`. (2cm)

**Publish object radius:**

If running in `mode:=actual`, to change radius of object:

In another terminal, run:
```bash
ros2 topic pub --once /perception/detected_object_radius std_msgs/msg/Float32 "{data: <object_radius>}"
```

For example, <object_radius> could be 0.07.

## Results
![Object Segmentation](https://raw.githubusercontent.com/Robots4Sustainability/perception/refs/heads/feat/table-segmentation/results/object_segmentation.png)
The above image is highlighting segmentation of the car parts which is then estimated to a sphere. This float sphere radius will be then published as msg to table_segemntation node, 
which will look at empty space and decide whether this sphere is possible to be placed on the table without colliding with other objects. 

Object Segemntation->Sphere Approximation->(passed the radius)->Table_segementation

**Place Object**

The results of running the table segmentation node will look like:

![alt text](place_object_1.png)

![alt text](place_object_2.png)

![alt text](no_place.png)
