import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class SimpleCameraNode(Node):
    def __init__(self):
        super().__init__('simple_camera_node')
        
        # Declare parameters
        self.declare_parameter('device_id', 0)
        self.declare_parameter('topic_name', '/camera/color/image_raw')
        
        device_id = self.get_parameter('device_id').get_parameter_value().integer_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        
        self.publisher = self.create_publisher(Image, topic_name, 10)
        self.bridge = CvBridge()
        
        self.get_logger().info(f"Opening camera device {device_id}...")
        self.cap = cv2.VideoCapture(device_id)
        
        self.use_synthetic = False
        if not self.cap.isOpened():
             self.get_logger().warn(f"Could not open camera device {device_id}! Switching to SYNTHETIC MODE (Test Pattern).")
             self.use_synthetic = True
             self.synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
             # Draw a "tool" (green rectangle) to detect
             cv2.rectangle(self.synthetic_frame, (200, 150), (440, 330), (0, 255, 0), -1) 

        timer_period = 0.03  # 30ms ~ 33 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        if self.use_synthetic:
            frame = self.synthetic_frame.copy()
            # Add some noise or movement?
            pass
        else:
            ret, frame = self.cap.read()
            if not ret:
                # self.get_logger().warn('Failed to capture frame')
                return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg() # Important for synchronization
        msg.header.frame_id = "camera_frame"
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleCameraNode()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
