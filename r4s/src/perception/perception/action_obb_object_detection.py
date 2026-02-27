import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path 

def get_package_name_from_path(file_path):
    """Dynamically find the package name from the file path."""
    p = Path(file_path)
    package_parts = p.parts[p.parts.index('site-packages') + 1:]
    return package_parts[0]

class OBBDetectorLifecycleNode(LifecycleNode):
    def __init__(self):
        self.package_name = get_package_name_from_path(__file__)
        super().__init__('obb_detector_node')
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('model_type', 'fine_tuned')    
        self.declare_parameter('input_mode', 'robot')  
        self.declare_parameter('model_path', '')           
        self.declare_parameter('conf_threshold', 0.6)      
        self.declare_parameter('device', '')               
        self.declare_parameter('class_names', [])          

        # Placeholders for resources
        self.model = None
        self.class_names = None
        self.image_sub = None
        self.annotated_image_pub = None
        self.detection_pub = None
        self.conf_threshold = 0.6

        self._is_active = False
        self.get_logger().info('YOLOv8-OBB Detector Lifecycle Node Initialized (Unconfigured).')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring OBB Detector Node...")
        
        try:
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
                package_share_directory = get_package_share_directory(self.package_name)
                if model_type == 'fine_tuned':
                    model_path = os.path.join(package_share_directory, 'models', 'screwdriver_obb_best.pt')
                else:
                    model_path = os.path.join(package_share_directory, 'models', 'yolov8n-obb.pt')

            self.get_logger().info(f"Using OBB model type '{model_type}' from: {model_path}")

            # Load Model
            self.model = YOLO(model_path)
            if device_param:
                try:
                    self.model.to(device_param)
                    self.get_logger().info(f"Model moved to device: {device_param}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to move model to device '{device_param}': {e}")

            # Optional override for class names
            if class_names_param:
                self.class_names = list(class_names_param)
                self.get_logger().info(f"Using class names from parameter (n={len(self.class_names)})")

            # Determine image topic
            if input_mode == 'robot':
                image_topic = '/camera/color/image_raw'
            elif input_mode == 'realsense':
                image_topic = '/camera/camera/color/image_raw'
            else:
                self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'realsense'")
                image_topic = '/camera/camera/color/image_raw'

            # Create Lifecycle Publishers
            self.annotated_image_pub = self.create_lifecycle_publisher(Image, '/annotated_image', 10)
            self.detection_pub = self.create_lifecycle_publisher(Detection2DArray, '/detections', 10)

            # Create Subscription
            self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Failed to configure OBB Detector Node: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating OBB Detector Node...")
        super().on_activate(state)
        self._is_active = True
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating OBB Detector Node...")
        self._is_active = False
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up OBB Detector resources...")
        
        self.destroy_publisher(self.annotated_image_pub)
        self.destroy_publisher(self.detection_pub)
        self.destroy_subscription(self.image_sub)

        self.annotated_image_pub = self.detection_pub = self.image_sub = None

        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down OBB Detector Node...")
        if self.model is not None:
            self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def image_callback(self, msg):
        if not self._is_active:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        # Run Inference
        results = self.model(cv_image)

        detection_array_msg = Detection2DArray()
        detection_array_msg.header = msg.header 

        for result in results:
            if not hasattr(result, 'obb') or result.obb is None:
                continue

            # Filter by confidence using boolean masking
            mask = result.obb.conf >= self.conf_threshold
            result.obb = result.obb[mask]

            # Publish Annotated Image
            annotated_image = result.plot()
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f'Annotated image conversion error: {e}')

            # Fill Detection2D Messages
            for obb in result.obb:
                detection_msg = Detection2D()
                detection_msg.header = msg.header 

                hypothesis = ObjectHypothesisWithPose()
                cls_id = int(obb.cls.item())
                
                if self.class_names and cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = self.model.names.get(cls_id, str(cls_id))

                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = float(obb.conf.item())
                detection_msg.results.append(hypothesis)

                # OBB Bounding Box (XYWHR)
                xywhr = obb.xywhr.cpu().numpy().flatten()
                
                detection_msg.bbox.center.position.x = float(xywhr[0])
                detection_msg.bbox.center.position.y = float(xywhr[1])
                detection_msg.bbox.center.theta = float(xywhr[4]) # Capture the rotation!
                
                detection_msg.bbox.size_x = float(xywhr[2])
                detection_msg.bbox.size_y = float(xywhr[3])

                detection_array_msg.detections.append(detection_msg)

        self.detection_pub.publish(detection_array_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OBBDetectorLifecycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()