import torch
import os

# Monkeypatch torch.load to disable weights_only enforcement by default
# This is necessary because PyTorch 2.6+ defaults weights_only=True, 
# but Ultralytics 8.1.0 (legacy req) relies on the old behavior for loading yolov8s.pt
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
         kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

from ultralytics import YOLO

def main():
    print("Starting YOLOv8 training with patched torch.load...")
    # Load model
    model = YOLO('yolov8s.pt')
    
    # Train
    # data path must be correct relative to cwd
    model.train(data='datasets/tools/data.yaml', epochs=50, imgsz=640)

if __name__ == '__main__':
    main()
