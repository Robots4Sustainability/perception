import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
import torch
import cv2
import os
from ament_index_python.packages import get_package_share_directory
from pathlib import Path 

from rclpy.signals import SignalHandlerOptions

def get_package_name_from_path(file_path):
    p = Path(file_path)
    package_parts = p.parts[p.parts.index('site-packages') + 1:]
    return package_parts[0]

class YoloDetectorLifecycleNode(LifecycleNode):
    def __init__(self):
        self.package_name = get_package_name_from_path(__file__)
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('model_type', 'fine_tuned')    
        self.declare_parameter('input_mode', 'robot')  
        self.declare_parameter('model_path', '')           
        self.declare_parameter('conf_threshold', 0.6)      
        self.declare_parameter('device', '')               
        self.declare_parameter('class_names', ["motor"])

        self.PICKABLE_CLASSES     = {'motor', 'motor_grip', 'speaker', 'unit'}
        self.NON_PICKABLE_CLASSES = {'assembly', 'enclosure', 'wire_plug'}          

        # Placeholders for resources
        self.model = None
        self.class_names = None
        self.image_sub = None
        self.annotated_image_pub = None
        self.pickable_image_pub = None
        self.detection_pub = None
        self.max_dim_pub = None
        self.conf_threshold = 0.6

        self._is_active = False
        self.get_logger().info('YOLO Detector Lifecycle Node Initialized (Unconfigured).')


    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring YOLO Node...")
        
        try:
            model_type = self.get_parameter('model_type').get_parameter_value().string_value
            input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
            explicit_model_path = self.get_parameter('model_path').get_parameter_value().string_value
            self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
            device_param = self.get_parameter('device').get_parameter_value().string_value
            class_names_param = self.get_parameter('class_names').get_parameter_value().string_array_value

            if explicit_model_path:
                model_path = explicit_model_path
            else:
                package_share_directory = get_package_share_directory(self.package_name)
                model_map = {
                    'fine_tuned': 'fine_tuned.pt',
                    'unit':'fine_tuned_old.pt',
                    'screw':      'screw_best.pt',
                    'default':    'yolov8n.pt'
                }
                model_filename = model_map.get(model_type, 'yolov8n.pt')
                model_path = os.path.join(package_share_directory, 'models', model_filename)

            self.get_logger().info(f"Using model type '{model_type}' from: {model_path}")

            self.model = YOLO(model_path)
            if device_param:
                try:
                    self.model.to(device_param)
                    self.get_logger().info(f"Model moved to device: {device_param}")
                except Exception as e:
                    self.get_logger().warn(f"Failed to move model to device '{device_param}': {e}")

            if class_names_param:
                self.class_names = list(class_names_param)

            if input_mode == 'robot':
                image_topic = '/camera/color/image_raw'
            else:
                image_topic = '/camera/camera/color/image_raw'

            # Lifecycle Publishers
            self.annotated_image_pub = self.create_lifecycle_publisher(Image, '/annotated_image', 10)
            self.pickable_image_pub  = self.create_lifecycle_publisher(Image, '/annotated_image/pickable', 10)
            self.detection_pub = self.create_lifecycle_publisher(Detection2DArray, '/detections', 10)
            self.max_dim_pub = self.create_lifecycle_publisher(Float32, '/detections/max_dimension', 10)

            # Subscriptions
            self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Failed to configure YOLO Node: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(f"Activating YOLO Node...")
        super().on_activate(state)
        self._is_active = True
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating YOLO Node...")
        self._is_active = False
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up YOLO resources...")
        
        self.destroy_publisher(self.annotated_image_pub)
        self.destroy_publisher(self.pickable_image_pub)
        self.destroy_publisher(self.detection_pub)
        self.destroy_publisher(self.max_dim_pub)
        self.destroy_subscription(self.image_sub)

        self.annotated_image_pub = self.pickable_image_pub = self.detection_pub = self.max_dim_pub = self.image_sub = None
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down YOLO Node...")
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
 
        results = self.model(cv_image, verbose=False)
        
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = msg.header 
 
        current_frame_max_dim = 0.0
 
        for result in results:
            filtered_boxes = [box for box in result.boxes if float(box.conf) >= self.conf_threshold]
 
            if filtered_boxes:
                box_data = torch.stack([b.data[0] for b in filtered_boxes])
                result.boxes = Boxes(box_data, orig_shape=result.orig_shape)
            else:
                result.boxes = Boxes(torch.empty((0, 6)), orig_shape=result.orig_shape)
 
            annotated_image = result.plot()
            try:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
                annotated_msg.header = msg.header
                self.annotated_image_pub.publish(annotated_msg)
            except Exception as e:
                self.get_logger().error(f'Annotated image conversion error: {e}')
 
            # --- PICKABLE / NON-PICKABLE OVERLAY ---
            pickable_image = cv_image.copy()
            for box in filtered_boxes:
                cls_id     = int(box.cls)
                class_name = (self.class_names[cls_id]
                              if self.class_names and cls_id < len(self.class_names)
                              else self.model.names.get(cls_id, str(cls_id)))
 
                if class_name in self.PICKABLE_CLASSES:
                    color = (0, 200, 0)        # green
                    label = f'{class_name} | PICKABLE'
                elif class_name in self.NON_PICKABLE_CLASSES:
                    color = (0, 0, 220)        # red
                    label = f'{class_name} | NON-PICKABLE'
                else:
                    color = (180, 180, 180)    # grey — unknown class
                    label = f'{class_name} | UNKNOWN'
 
                xyxy = box.xyxy.cpu().numpy().flatten().astype(int)
                x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                conf_str = f'{float(box.conf):.2f}'
 
                # Box
                cv2.rectangle(pickable_image, (x1, y1), (x2, y2), color, 2)
 
                # Label background + text
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness  = 1
                full_label = f'{label} {conf_str}'
                (tw, th), baseline = cv2.getTextSize(full_label, font, font_scale, thickness)
                cv2.rectangle(pickable_image, (x1, y1 - th - baseline - 4), (x1 + tw + 2, y1), color, -1)
                cv2.putText(pickable_image, full_label,
                            (x1 + 1, y1 - baseline - 2),
                            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
 
            try:
                pickable_msg = self.bridge.cv2_to_imgmsg(pickable_image, encoding='bgr8')
                pickable_msg.header = msg.header
                self.pickable_image_pub.publish(pickable_msg)
            except Exception as e:
                self.get_logger().error(f'Pickable image conversion error: {e}')
 
            for box in filtered_boxes:
                detection_msg = Detection2D()
                detection_msg.header = msg.header
 
                hypothesis = ObjectHypothesisWithPose()
                cls_id = int(box.cls)
                
                if self.class_names and cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = self.model.names.get(cls_id, str(cls_id))
 
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = float(box.conf)
                detection_msg.results.append(hypothesis)
 
                xywh = box.xywh.cpu().numpy().flatten()
                w = float(xywh[2])
                h = float(xywh[3])
                detection_msg.bbox.center.position.x = float(xywh[0])
                detection_msg.bbox.center.position.y = float(xywh[1])
                detection_msg.bbox.size_x = w
                detection_msg.bbox.size_y = h
 
                # Track the largest dimension
                max_dim = max(w, h)
                if max_dim > current_frame_max_dim:
                    current_frame_max_dim = max_dim
 
                detection_array_msg.detections.append(detection_msg)
 
        self.detection_pub.publish(detection_array_msg)
 
        if detection_array_msg.detections:
            dim_msg = Float32()
            dim_msg.data = current_frame_max_dim
            self.max_dim_pub.publish(dim_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorLifecycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
