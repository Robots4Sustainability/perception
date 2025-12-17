## Tool Detection Model Training

We use a YOLOv8-small model trained on the specific tools dataset.

### Training Command
```bash
yolo task=detect mode=train model=yolov8s.pt data=datasets/tools/data.yaml epochs=50 imgsz=640
```

### Metrics
*   **mAP@0.5**: 0.9813 (Target: > 0.95)
*   **Precision**: 0.9907
*   **Recall**: 0.9755

To run the trained model:
```bash
ros2 launch perception tool_pipeline.launch.py model_path:=models/tools.pt
```
