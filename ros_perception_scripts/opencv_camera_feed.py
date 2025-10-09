import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class SimpleCameraNode(Node):
    def __init__(self):
        super().__init__('simple_camera_node')
        self.publisher = self.create_publisher(Image, 'image_raw', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(4) # Video4Linux device

        timer_period = 0.03  # 30ms ~ 33 FPS
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
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
