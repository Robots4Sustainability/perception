import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct

class TableSegmentationNode(Node):
    def __init__(self):
        super().__init__('table_segmentation_node')

        # Declare Parameters
        self.declare_parameter('input_mode', 'robot')
        self.declare_parameter('debug_viz', True)

        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.debug_viz = self.get_parameter('debug_viz').get_parameter_value().bool_value

        if input_mode == 'robot':
            pc_topic = '/camera/depth/color/points'
        elif input_mode == 'realsense':
            pc_topic = '/camera/camera/depth/color/points'
        elif input_mode == 'cropped':
            pc_topic = '/cropped_pointcloud'
        else:
            pc_topic = '/camera/depth/color/points'

        self.get_logger().info(f"Subscribing to: {pc_topic}")

        self.pc_sub = self.create_subscription(PointCloud2, pc_topic, self.cloud_callback, 10)
        self.place_pose_pub = self.create_publisher(PoseStamped, '/perception/target_place_pose', 10)
        self.table_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/table_plane', 10)
        self.object_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/objects', 10)

    def cloud_callback(self, ros_cloud_msg):
        self.get_logger().info(f"Processing cloud: {ros_cloud_msg.width}x{ros_cloud_msg.height}")

        pcd = self.convert_ros_to_o3d(ros_cloud_msg)
        if pcd is None:
            return

        pcd_down = pcd.voxel_down_sample(voxel_size=0.005)

        try:
            plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.02,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        except Exception as e:
            self.get_logger().warn("Could not find a plane (table) in view.")
            return

        table_cloud = pcd_down.select_by_index(inliers)
        object_cloud = pcd_down.select_by_index(inliers, invert=True)

        target_point = self.find_empty_spot(table_cloud, object_cloud)
        
        if target_point is not None:
            self.publish_target_pose(target_point[0], target_point[1], target_point[2], ros_cloud_msg.header)
        else:
            self.get_logger().warn("Table full or no safe spot found!")

        if self.debug_viz:
            # Paint Green and Red
            table_cloud.paint_uniform_color([0, 1, 0])
            object_cloud.paint_uniform_color([1, 0, 0])
            
            # Publish with Color support
            self.publish_o3d(table_cloud, self.table_cloud_pub, ros_cloud_msg.header)
            self.publish_o3d(object_cloud, self.object_cloud_pub, ros_cloud_msg.header)

    def find_empty_spot(self, table_cloud, object_cloud):
        if len(table_cloud.points) == 0:
            return None

        # KDTree for fast distance check
        table_tree = o3d.geometry.KDTreeFlann(table_cloud)
        
        has_objects = len(object_cloud.points) > 0
        if has_objects:
            object_tree = o3d.geometry.KDTreeFlann(object_cloud)
        
        table_pts = np.asarray(table_cloud.points)
        min_x, min_y = np.min(table_pts[:,0]), np.min(table_pts[:,1])
        max_x, max_y = np.max(table_pts[:,0]), np.max(table_pts[:,1])
        avg_z = np.mean(table_pts[:,2])

        # Grid search parameters
        step_size = 0.05 # Check every 5cm
        safety_radius = 0.15 # 15cm from objects
        
        best_point = None
        max_dist_to_obj = -1.0

        for x in np.arange(min_x, max_x, step_size):
            for y in np.arange(min_y, max_y, step_size):
                candidate = np.array([x, y, avg_z])
                
                [k, _, _] = table_tree.search_radius_vector_3d(candidate, 0.05)
                if k == 0: continue 

                if has_objects:
                    [_, _, dist_sq] = object_tree.search_knn_vector_3d(candidate, 1)
                    dist = np.sqrt(dist_sq[0])
                    if dist < safety_radius:
                        continue 
                    
                    # the spot furthest from everything
                    if dist > max_dist_to_obj:
                        max_dist_to_obj = dist
                        best_point = candidate
                else:
                    return candidate # Empty table, take first valid spot

        return best_point

    def convert_ros_to_o3d(self, ros_cloud):
        try:
            pcd_as_numpy = np.array([
                [x, y, z] 
                for x, y, z in pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True)
            ])
            
            if pcd_as_numpy.shape[0] == 0:
                return None

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_as_numpy)
            return pcd
        except Exception as e:
            self.get_logger().error(f'Conversion error: {e}')
            return None

    def publish_target_pose(self, x, y, z, header):
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = z
        pose_msg.pose.orientation.w = 1.0 
        self.place_pose_pub.publish(pose_msg)

    def publish_o3d(self, o3d_cloud, publisher, header):
        points = np.asarray(o3d_cloud.points)
        if len(points) == 0:
            return

        # If no colors, just send XYZ
        if not o3d_cloud.has_colors():
            msg = pc2.create_cloud_xyz32(header, points)
            publisher.publish(msg)
            return

        # Pack RGB color into the message
        colors = np.asarray(o3d_cloud.colors)
        points_with_color = []
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i]
            rgb_int = (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
            rgb_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
            points_with_color.append([x, y, z, rgb_float])

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

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
