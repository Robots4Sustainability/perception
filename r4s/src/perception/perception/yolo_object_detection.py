import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import torch
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path 

def get_package_name_from_path(file_path):
    """Dynamically find the package name from the file path."""
    # e.g., /path/to/ws/install/my_pkg/lib/pythonX.Y/site-packages/my_pkg/node.py
    # This will return 'my_pkg'
    p = Path(file_path)
    # The package name is the directory name after 'site-packages'
    # parts will be like: ('/', 'path', ..., 'site-packages', 'my_pkg', 'node.py')
    package_parts = p.parts[p.parts.index('site-packages') + 1:]
    return package_parts[0]


class YoloDetectorNode(Node):
    def __init__(self):
        self.package_name = get_package_name_from_path(__file__)
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()

        # Declare and get parameters
        self.declare_parameter('model_type', 'fine_tuned')    # 'default' or 'fine_tuned'
        self.declare_parameter('input_mode', 'robot')  # 'robot' or 'realsense'
        self.declare_parameter('model_path', '')           # Optional explicit path to .pt
        self.declare_parameter('conf_threshold', 0.6)      # Confidence threshold for filtering
        self.declare_parameter('device', '')               # e.g., 'cpu', 'cuda:0' (optional)
        self.declare_parameter('class_names', [])          # Optional override list for class names

        model_type = self.get_parameter('model_type').get_parameter_value().string_value
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        explicit_model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        device_param = self.get_parameter('device').get_parameter_value().string_value
        class_names_param = self.get_parameter('class_names').get_parameter_value().string_array_value

        # Determine model path
        if explicit_model_path:
            model_path = explicit_model_path
        else:
            # Get the absolute path to the package's share directory
            package_share_directory = get_package_share_directory(self.package_name)
            
            if model_type == 'fine_tuned':
                # Construct the path to the model relative to the share directory
                model_path = os.path.join(package_share_directory, 'models', 'fine_tuned.pt')
            else:
                model_path = os.path.join(package_share_directory, 'models', 'yolov8n.pt')

        self.get_logger().info(f"Using model type '{model_type}' from: {model_path}")

        try:
            if device_param:
                self.model = YOLO(model_path)
                # ultralytics model allows .to(device) for device selection
                try:
                    self.model.to(device_param)
                    self.get_logger().info(f"Model moved to device: {device_param}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to move model to device '{device_param}': {e}")
            else:
                self.model = YOLO(model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise

        # Optional override for class names
        self.class_names = None
        try:
            if class_names_param:
                self.class_names = list(class_names_param)
                self.get_logger().info(f"Using class names from parameter (n={len(self.class_names)})")
        except Exception as e:
            self.get_logger().warn(f"Failed to read class_names parameter: {e}")

        # Determine image topic
        if input_mode == 'robot':
            image_topic = '/camera/color/image_raw'
        elif input_mode == 'realsense':
            image_topic = '/camera/camera/color/image_raw'
        else:
            self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'realsense'")
            image_topic = '/camera/camera/color/image_raw'

        self.get_logger().info(f"Subscribing to image topic: {image_topic}")

        # Create subscriptions and publishers
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)

        self.get_logger().info('YOLOv8 Detector Node started.')

    def image_callback(self, msg):
        try:
            # 1. Convert Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        # 2. Run Inference
        results = self.model(cv_image)

        # 3. Initialize Message with ORIGINAL Timestamp
        # This is critical for the PointCloudCropper's Synchronizer
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = msg.header 

        for result in results:
            # Filter by confidence
            filtered_boxes = [box for box in result.boxes if float(box.conf) >= self.conf_threshold]

            # Update boxes for result.plot()
            if filtered_boxes:
                box_data = torch.stack([b.data[0] for b in filtered_boxes])
                result.boxes = Boxes(box_data, orig_shape=result.orig_shape)
            else:
                result.boxes = Boxes(torch.empty((0, 6)), orig_shape=result.orig_shape)

            # Publish Annotated Image (using original timestamp)
            annotated_image = result.plot()
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f'Annotated image conversion error: {e}')

            # 4. Fill Detection2D Messages
            for box in filtered_boxes:
                detection_msg = Detection2D()
                detection_msg.header = msg.header # Consistency across sub-messages

                hypothesis = ObjectHypothesisWithPose()
                cls_id = int(box.cls)
                
                # Safe class name lookup
                if self.class_names and cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = self.model.names.get(cls_id, str(cls_id))

                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = float(box.conf)
                detection_msg.results.append(hypothesis)

                # Bounding Box (XYWH)
                xywh = box.xywh.cpu().numpy().flatten()
                detection_msg.bbox.center.position.x = float(xywh[0])
                detection_msg.bbox.center.position.y = float(xywh[1])
                detection_msg.bbox.size_x = float(xywh[2])
                detection_msg.bbox.size_y = float(xywh[3])

                detection_array_msg.detections.append(detection_msg)

        # 5. Single Publish per Image Frame
        self.detection_pub.publish(detection_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()