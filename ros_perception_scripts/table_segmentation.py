import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct

class TableSegmentationNode(Node):
    def __init__(self):
        super().__init__('table_segmentation_node')

        self.declare_parameter('input_mode', 'realsense')  # 'robot' or 'realsense' or 'cropped'
        self.declare_parameter('debug_viz', True)          # Publish colored clouds for RViz?

        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.debug_viz = self.get_parameter('debug_viz').get_parameter_value().bool_value

        # Determine Point Cloud Topic
        if input_mode == 'robot':
            pc_topic = '/kinova/depth/color/points'
        elif input_mode == 'realsense':
            pc_topic = '/camera/camera/depth/color/points'
        elif input_mode == 'cropped':
            pc_topic = '/cropped_pointcloud'
        else:
            self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'cropped'")
            pc_topic = '/cropped_pointcloud'

        self.get_logger().info(f"Subscribing to PointCloud topic: {pc_topic}")

        # Subscribers and Publishers
        self.pc_sub = self.create_subscription(PointCloud2, pc_topic, self.cloud_callback, 10)
        
        # Publisher for the target placement pose (X, Y, Z + Orientation)
        self.place_pose_pub = self.create_publisher(PoseStamped, '/perception/target_place_pose', 10)
        
        # Debug Publishers (visualize what the robot thinks is the table vs objects)
        self.table_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/table_plane', 10)
        self.object_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/objects', 10)

    def cloud_callback(self, ros_cloud_msg):
        # Convert ROS PointCloud2 to Open3D
        pcd = self.convert_ros_to_o3d(ros_cloud_msg)
        if pcd is None or len(pcd.points) < 100:
            return # Not enough data

        # Pre-processing (Downsample for speed) - Reduces points to 1 every 5mm
        pcd_down = pcd.voxel_down_sample(voxel_size=0.005)

        # Plane Segmentation (using RANSAC)
        # Points within distance_threshold of the plane are considered table
        # ransac_n: 3 points define a plane
        plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.015,
                                                      ransac_n=3,
                                                      num_iterations=1000)

        # Separate table and objects
        table_cloud = pcd_down.select_by_index(inliers)
        object_cloud = pcd_down.select_by_index(inliers, invert=True)

        # Find empty space - Find the average center of the table.
        # (later do grid search if objects are in the middle)
        
        if len(table_cloud.points) > 0:
            table_points = np.asarray(table_cloud.points)
            center_x = np.mean(table_points[:, 0])
            center_y = np.mean(table_points[:, 1])
            center_z = np.mean(table_points[:, 2])

            # Publish the Target Pose
            self.publish_target_pose(center_x, center_y, center_z, ros_cloud_msg.header)

        # Publish Visualization
        if self.debug_viz:
            # Paint table Green, Objects Red
            table_cloud.paint_uniform_color([0, 1, 0])
            object_cloud.paint_uniform_color([1, 0, 0])
            
            self.publish_o3d(table_cloud, self.table_cloud_pub, ros_cloud_msg.header)
            self.publish_o3d(object_cloud, self.object_cloud_pub, ros_cloud_msg.header)

    def publish_target_pose(self, x, y, z, header):
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        
        # might need to change this
        pose_msg.pose.orientation.w = 1.0
        
        self.place_pose_pub.publish(pose_msg)

    def convert_ros_to_o3d(self, ros_cloud):
        # Read x,y,z from ROS message
        try:
            pcd_as_numpy = np.array(list(pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True)))
            if pcd_as_numpy.shape[0] == 0:
                return None
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_as_numpy)
            return pcd
        except Exception as e:
            self.get_logger().error(f'Conversion error: {e}')
            return None

def publish_o3d(self, o3d_cloud, publisher, header):
        # Check if points exist
        points = np.asarray(o3d_cloud.points)
        if len(points) == 0:
            return

        if not o3d_cloud.has_colors():
            # Fallback to simple XYZ if no colors exist
            msg = pc2.create_cloud_xyz32(header, points)
            publisher.publish(msg)
            return

        # Pack data into (x, y, z, rgb)
        # Open3D stores colors as floats [0.0 - 1.0], ROS needs packed int
        colors = np.asarray(o3d_cloud.colors)
        points_with_color = []
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            
            rgb_int = (r << 16) | (g << 8) | b
            rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
            points_with_color.append([x, y, z, rgb_float])

        # Define the fields (x, y, z, rgb)
        fields = [
            pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
            pc2.PointField(name='rgb', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
        ]

        # Create and publish message
        msg = pc2.create_cloud(header, fields, points_with_color)
        publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TableSegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()