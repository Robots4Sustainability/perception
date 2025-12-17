import torch
import cv2
from ultralytics import YOLO

# Monkeypatch torch.load to disable weights_only enforcement
# This is necessary because we are using PyTorch 2.6+ with Ultralytics 8.1.0
if hasattr(torch, 'load'):
    original_load = torch.load
    def safe_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
             kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = safe_load

def main():
    # Load the trained model
    model_path = 'models/tools.pt'
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference on the webcam (source=0)
    # show=True opens a window with results
    # conf=0.6 sets confidence threshold
    print("Starting webcam... Press 'q' to exit (if window allows) or Ctrl+C in terminal.")
    results = model.predict(source='0', show=True, conf=0.6, stream=True)
    
    # We iterate over the stream to keep the script running
    for r in results:
        pass # The show=True handles the display

if __name__ == '__main__':
    main()
