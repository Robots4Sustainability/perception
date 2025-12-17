### Running Table Height
```code
ros2 run table_height_predictor table_heigt
```
### Running Floor Detection
```
ros2 run table_height_predictor detect_floor
```
Although this node isnt useful to the table height but, it can help us to separate objects from the floor. This could be useful if one wishes to only
segment the floor. Moreover, it also helps to estimate the angle of depression of the robot's camera.


## Results
![Table Height Predictor](https://raw.githubusercontent.com/Robots4Sustainability/perception/refs/heads/feat/table-height/results/Table_Height.png)
- It uses Yolo World Model with SAM (Segmentation Anything Meta) using Zero Shot learning and prompt like white "standing desk" to segment the table.

- It then uses RANSAC algorithim to identify the plane of the floor. It then draws a line between the line between the floor plane and the top of the 
desk. 

- It has an error rate of +-5cm to the actual height so it should be used with a safety margin. In the example above the actual height was 58cm.
- This can work with varying heights of the table.



