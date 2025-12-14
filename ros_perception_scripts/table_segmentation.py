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
        self.viz_pose_pub = self.create_publisher(PoseStamped, '/perception/debug/viz_pose', 10) # For RViz - Visual Offset above the table

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
            quat, arrow_vector = self.get_orientation_from_plane(plane_model)

            self.get_logger().info(
                f"Target: X={target_point[0]:.2f}, Y={target_point[1]:.2f}, Z={target_point[2]:.2f}"
            )

            # Publish the real target pose for the robot
            real_pose = PoseStamped()
            real_pose.header = ros_cloud_msg.header
            real_pose.pose.position.x = float(target_point[0])
            real_pose.pose.position.y = float(target_point[1])
            real_pose.pose.position.z = float(target_point[2])
            real_pose.pose.orientation.x = float(quat[0])
            real_pose.pose.orientation.y = float(quat[1])
            real_pose.pose.orientation.z = float(quat[2])
            real_pose.pose.orientation.w = float(quat[3])
            
            self.place_pose_pub.publish(real_pose)

            # Publish a visualization pose slightly above the table for RViz
            viz_len = 0.15
            hover_point = target_point - (arrow_vector * viz_len)
            
            viz_pose = PoseStamped()
            viz_pose.header = ros_cloud_msg.header
            viz_pose.pose.position.x = float(hover_point[0])
            viz_pose.pose.position.y = float(hover_point[1])
            viz_pose.pose.position.z = float(hover_point[2])
            viz_pose.pose.orientation = real_pose.pose.orientation 
            
            self.viz_pose_pub.publish(viz_pose)
        else:
            self.get_logger().warn("Table full or no safe spot found!")

        if self.debug_viz:
            # Paint Green and Red
            table_cloud.paint_uniform_color([0, 1, 0])
            object_cloud.paint_uniform_color([1, 0, 0])
            
            # Publish with Color support
            self.publish_o3d(table_cloud, self.table_cloud_pub, ros_cloud_msg.header)
            self.publish_o3d(object_cloud, self.object_cloud_pub, ros_cloud_msg.header)
    
    def get_orientation_from_plane(self, plane_model):
        """
        For visualization,
        Calculates a quaternion where the X-AXIS (The Red Arrow) points into the table.
        """
        # Get the Normal Vector [a,b,c]
        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)

        # Define the arrow direction (X-axis)
        if normal[2] < 0:
            target_x = normal 
        else:
            target_x = -normal

        # Construct Orthogonal Axes
        ref_vector = np.array([0, 1, 0])
        
        # If too close to parallel, use World X (1,0,0)
        if np.abs(np.dot(target_x, ref_vector)) > 0.9:
            ref_vector = np.array([1, 0, 0])

        # Z = X cross Ref
        z_axis = np.cross(target_x, ref_vector)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # Y = Z cross X (Ensure orthogonality)
        y_axis = np.cross(z_axis, target_x)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Create Rotation Matrix [ X  Y  Z ]
        R = np.array([target_x, y_axis, z_axis]).T

        # Convert to Quaternion [x, y, z, w]
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S 
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            qw = (R[2,1] - R[1,2]) / S
            qx = 0.25 * S
            qy = (R[0,1] + R[1,0]) / S
            qz = (R[0,2] + R[2,0]) / S
        elif (R[1,1] > R[2,2]):
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            qw = (R[0,2] - R[2,0]) / S
            qx = (R[0,1] + R[1,0]) / S
            qy = 0.25 * S
            qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            qw = (R[1,0] - R[0,1]) / S
            qx = (R[0,2] + R[2,0]) / S
            qy = (R[1,2] + R[2,1]) / S
            qz = 0.25 * S

        return [qx, qy, qz, qw], target_x

    def find_empty_spot(self, table_cloud, object_cloud):
        if len(table_cloud.points) == 0:
            return None

        # KDTree for fast distance check
        table_tree = o3d.geometry.KDTreeFlann(table_cloud)
        
        # Check distance to objects
        has_objects = len(object_cloud.points) > 0
        if has_objects:
            object_tree = o3d.geometry.KDTreeFlann(object_cloud)
        
        # Calculate table boundaries and center
        table_pts = np.asarray(table_cloud.points)
        min_x, min_y = np.min(table_pts[:,0]), np.min(table_pts[:,1])
        max_x, max_y = np.max(table_pts[:,0]), np.max(table_pts[:,1])
        avg_z = np.mean(table_pts[:,2])
        
        center_x = np.mean(table_pts[:,0])
        center_y = np.mean(table_pts[:,1])

        step_size = 0.05 # Check every 5cm
        safety_radius = 0.05 # 5cm from objects
        
        # Ignore the outer 2.5cm of the bounding box
        edge_margin = 0.025
        
        best_point = None
        best_score = -float('inf')

        # Iterate through the grid by also skipping the edges
        for x in np.arange(min_x + edge_margin, max_x - edge_margin, step_size):
            for y in np.arange(min_y + edge_margin, max_y - edge_margin, step_size):
                candidate = np.array([x, y, avg_z])
                
                # Check if there is a table surface here
                # If 0 neighbors, it's a hole or off-shape edge.
                [k, _, _] = table_tree.search_radius_vector_3d(candidate, 0.05)
                if k == 0: continue 

                # Calculate distance to center
                dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

                # Avoid objects
                if has_objects:
                    [_, _, dist_sq] = object_tree.search_knn_vector_3d(candidate, 1)
                    dist_to_obj = np.sqrt(dist_sq[0])
                    
                    if dist_to_obj < safety_radius:
                        continue # Too close to an object
                    
                    # Calculate score: favor distance from objects and closeness to center
                    score = dist_to_obj - (0.8 * dist_to_center)
                else:
                    # If table is empty, find the spot closest to center
                    score = -dist_to_center

                # Pick the best score
                if score > best_score:
                    best_score = score
                    best_point = candidate

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
