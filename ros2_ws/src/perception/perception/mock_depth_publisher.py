import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2

class MockDepthPublisher(Node):
    def __init__(self):
        super().__init__('mock_depth_publisher')
        self.bridge = CvBridge()
        
        # Subscribe to webcam image
        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        
        # Publish mock pointcloud
        self.pc_pub = self.create_publisher(PointCloud2, '/camera/depth/color/points', 10)
        
        self.get_logger().info("Mock Depth Publisher Started. Generating planar depth at Z=1.0m")

    def image_callback(self, img_msg):
        # Create a simple planar pointcloud matching the image resolution
        height = img_msg.height
        width = img_msg.width
        
        # Generate grid of X, Y coordinates
        # Simplified intrinsic model: assume center is (w/2, h/2) and focal length ~ width
        fx, fy = width, width
        cx, cy = width / 2, height / 2
        
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        z = np.ones_like(u) * 1.0  # Fixed depth of 1 meter
        
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack into (N, 3) array
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Create header
        header = img_msg.header
        
        # Create PointCloud2
        pc_msg = pc2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(pc_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MockDepthPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
