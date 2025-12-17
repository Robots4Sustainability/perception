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
from std_srvs.srv import SetBool # Required for the toggle service

def get_package_name_from_path(file_path):
    """Dynamically find the package name from the file path."""
    p = Path(file_path)
    try:
        package_parts = p.parts[p.parts.index('site-packages') + 1:]
        return package_parts[0]
    except ValueError:
        # Fallback if not installed in site-packages (e.g. running from source)
        return 'perception' 

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.package_name = get_package_name_from_path(__file__)
        self.bridge = CvBridge()

        # --- PARAMETERS ---
        self.declare_parameter('model_type', 'default')
        self.declare_parameter('input_mode', 'realsense')
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('device', '')
        self.declare_parameter('class_names', [])

        model_type = self.get_parameter('model_type').get_parameter_value().string_value
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        explicit_model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        device_param = self.get_parameter('device').get_parameter_value().string_value
        class_names_param = self.get_parameter('class_names').get_parameter_value().string_array_value

        # --- MODEL PATH LOGIC ---
        if explicit_model_path:
            model_path = explicit_model_path
        else:
            try:
                package_share_directory = get_package_share_directory(self.package_name)
                if model_type == 'fine_tuned':
                    model_path = os.path.join(package_share_directory, 'models', 'fine_tuned.pt')
                else:
                    model_path = os.path.join(package_share_directory, 'models', 'yolov8n.pt')
            except Exception:
                # Fallback for when running locally/testing without full install
                model_path = 'yolov8n.pt' 
                self.get_logger().warn(f"Could not find package share directory. Defaulting model path to: {model_path}")

        self.get_logger().info(f"Using model type '{model_type}' from: {model_path}")

        # --- 1. LOAD MODEL (Heavy operation, done ONCE at startup) ---
        self.get_logger().info("Loading YOLO model... (This stays in RAM)")
        try:
            if device_param:
                self.model = YOLO(model_path)
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
        if class_names_param:
            self.class_names = list(class_names_param)

        # --- 2. DETERMINE TOPIC (Save for later) ---
        if input_mode == 'robot':
            self.image_topic = '/camera/color/image_raw'
        elif input_mode == 'realsense':
            self.image_topic = '/camera/camera/color/image_raw'
        else:
            self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'realsense'")
            self.image_topic = '/camera/camera/color/image_raw'

        # --- 3. SETUP PUBLISHERS ---
        self.annotated_image_pub = self.create_publisher(Image, '/annotated_image', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)

        # --- 4. OPTIMIZATION / SOFT LIFECYCLE ---
        # We do NOT subscribe here. We start in "Standby" mode.
        self.image_sub = None 
        
        # Create the service to wake up the node
        self.srv = self.create_service(SetBool, 'toggle_yolo', self.toggle_callback)

        self.get_logger().info(f"YOLOv8 Node Initialized in STANDBY mode.")
        self.get_logger().info(f"Send 'True' to service '/toggle_yolo' to start processing {self.image_topic}")

    def toggle_callback(self, request, response):
        """Service callback to Turn processing ON or OFF"""
        if request.data: # REQUEST: ENABLE
            if self.image_sub is None:
                self.get_logger().info(f"ACTIVATING: Subscribing to {self.image_topic}...")
                self.image_sub = self.create_subscription(
                    Image, self.image_topic, self.image_callback, 10
                )
                response.message = "YOLO Activated"
                response.success = True
            else:
                response.message = "Already Active"
                response.success = True
        else: # REQUEST: DISABLE
            if self.image_sub is not None:
                self.get_logger().info("DEACTIVATING: Unsubscribing to save CPU...")
                self.destroy_subscription(self.image_sub)
                self.image_sub = None
                response.message = "YOLO Deactivated"
                response.success = True
            else:
                response.message = "Already Inactive"
                response.success = True
        return response

    def image_callback(self, msg):
        """This function ONLY runs when the node is Activated"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        # Inference
        results = self.model(cv_image, verbose=False) # verbose=False keeps terminal clean
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = msg.header

        for result in results:
            filtered_boxes = [box for box in result.boxes if float(box.conf) >= self.conf_threshold]

            # 1. Publish Annotated Image
            annotated_image = result.plot()
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f'Annotated image conversion error: {e}')

            # 2. Publish Detections
            for box in filtered_boxes:
                detection_msg = Detection2D()
                detection_msg.header = msg.header

                hypothesis = ObjectHypothesisWithPose()
                try:
                    if self.class_names and int(box.cls) < len(self.class_names):
                        class_name = self.class_names[int(box.cls)]
                    else:
                        class_name = self.model.names[int(box.cls)]
                except Exception:
                    class_name = str(int(box.cls))
                
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = float(box.conf)
                detection_msg.results.append(hypothesis)

                xywh = box.xywh.cpu().numpy().flatten()
                detection_msg.bbox.center.position.x = float(xywh[0])
                detection_msg.bbox.center.position.y = float(xywh[1])
                detection_msg.bbox.size_x = float(xywh[2])
                detection_msg.bbox.size_y = float(xywh[3])

                detection_array_msg.detections.append(detection_msg)

        self.detection_pub.publish(detection_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()