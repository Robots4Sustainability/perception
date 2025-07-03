import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os

class YOLOv8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')
        self.bridge = CvBridge()
        model_path = os.path.expanduser('~/ros2_ws/models/yolov8n.pt')
        self.model = YOLO(model_path)  # Replace with your model if custom

        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(Image, '/yolov8/image_annotated', 10)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(frame)
        annotated_frame = results[0].plot()
        out_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        self.publisher.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
