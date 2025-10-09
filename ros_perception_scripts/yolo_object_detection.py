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


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        self.bridge = CvBridge()

        # Declare and get parameters
        self.declare_parameter('model_type', 'default')    # 'default' or 'fine_tuned'
        self.declare_parameter('input_mode', 'realsense')  # 'robot' or 'realsense'

        model_type = self.get_parameter('model_type').get_parameter_value().string_value
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value

        # Determine model path
        if model_type == 'fine_tuned':
            model_path = '/home/mohsin/official_build/ros2_ws/models/fine_tuned.pt'
        else:
            model_path = '/home/mohsin/official_build/ros2_ws/models/yolov8n.pt'

        self.get_logger().info(f"Using model type '{model_type}' from: {model_path}")
        self.model = YOLO(model_path)

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

        self.conf_threshold = 0.6  # Confidence threshold for filtering

        self.get_logger().info('YOLOv8 Detector Node started.')

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Image conversion error: {e}')
            return

        results = self.model(cv_image)
        detection_array_msg = Detection2DArray()
        detection_array_msg.header = msg.header

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

            for box in filtered_boxes:
                detection_msg = Detection2D()
                detection_msg.header = msg.header

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = self.model.names[int(box.cls)]
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
