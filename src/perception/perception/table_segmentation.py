import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32, Header
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

        self.declare_parameter('mode', 'test')            # 'test' or 'actual'
        self.declare_parameter('test_radius', 0.07)       # Default radius for test mode (7cm)
        self.declare_parameter('safety_margin', 0.02)     # Default safety margin (2cm)

        # Get Parameters
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.debug_viz = self.get_parameter('debug_viz').get_parameter_value().bool_value
        self.op_mode = self.get_parameter('mode').get_parameter_value().string_value
        self.current_radius = self.get_parameter('test_radius').get_parameter_value().double_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value

        self.get_logger().info(f"Starting in MODE: {self.op_mode}")
        self.get_logger().info(f"Initial Radius: {self.current_radius}m | Safety Margin: {self.safety_margin}m")

        if input_mode == 'robot':
            pc_topic = '/camera/depth/color/points'
        elif input_mode == 'realsense':
            pc_topic = '/camera/camera/depth/color/points'
        else:
            pc_topic = '/camera/depth/color/points'

        self.get_logger().info(f"Subscribing to: {pc_topic}")

        self.pc_sub = self.create_subscription(PointCloud2, pc_topic, self.cloud_callback, 10)

        # Radius Subscriber (Only active in actual mode)
        if self.op_mode == 'actual':
            self.radius_sub = self.create_subscription(
                Float32, 
                '/perception/detected_object_radius', 
                self.radius_callback, 
                10
            )
            self.get_logger().info("Listening to /perception/detected_object_radius for updates...")

        # Publishers
        self.place_pose_pub = self.create_publisher(PoseStamped, '/perception/target_place_pose', 10)
        self.viz_sphere_pub = self.create_publisher(Marker, '/perception/debug/viz_sphere', 10) # Publisher for the Safety Sphere

        # Debug Publishers
        self.table_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/table_plane', 10)
        self.object_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/objects', 10)
        self.viz_pub = self.create_publisher(Marker, '/perception/debug/viz_arrow', 10) # For RViz - Visual Offset above the table
    
    def radius_callback(self, msg):
        """Called only in actual mode when a new object size is detected."""
        if msg.data > 0.01:  # Minimum 1cm radius
            self.current_radius = float(msg.data)
            self.get_logger().info(f"Updated Object Radius to: {self.current_radius:.3f}m")

    def cloud_callback(self, ros_cloud_msg):
        self.get_logger().info(f"Processing cloud: {ros_cloud_msg.width}x{ros_cloud_msg.height}")

        pcd = self.convert_ros_to_o3d(ros_cloud_msg)
        if pcd is None:
            return
        
        max_depth = 1.2  # 1.2 meters
        
        points = np.asarray(pcd.points)
        if len(points) > 0:
            # Keep points where Z is less than max_depth
            mask = points[:, 2] < max_depth
            pcd = pcd.select_by_index(np.where(mask)[0])
        
        if len(pcd.points) < 100:
            self.get_logger().warn("No points left after depth filtering!")
            return

        pcd_down = pcd.voxel_down_sample(voxel_size=0.005)

        try:
            plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.008,
                                                        ransac_n=3,
                                                        num_iterations=1000)
        except Exception as e:
            self.get_logger().warn("Could not find a plane (table) in view.")
            return

        table_cloud = pcd_down.select_by_index(inliers)
        raw_object_cloud = pcd_down.select_by_index(inliers, invert=True)

        # Filter objects above the table
        object_cloud = self.filter_objects_above_table(raw_object_cloud, plane_model)

        total_check_radius = self.current_radius + self.safety_margin
        target_point = self.find_empty_spot(table_cloud, object_cloud, total_check_radius, plane_model)
        
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

            self.publish_arrow_marker(viz_pose)

            self.publish_safety_sphere(real_pose, self.current_radius)
        else:
            self.get_logger().warn("Table full or no safe spot found!")

        if self.debug_viz:
            # Paint Green and Red
            table_cloud.paint_uniform_color([0, 1, 0])
            object_cloud.paint_uniform_color([1, 0, 0])
            
            # Publish with Color support
            self.publish_o3d(table_cloud, self.table_cloud_pub, ros_cloud_msg.header)
            self.publish_o3d(object_cloud, self.object_cloud_pub, ros_cloud_msg.header)

    def publish_safety_sphere(self, pose_stamped, radius):
        marker = Marker()
        marker.header = pose_stamped.header
        marker.ns = "safety_sphere"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = pose_stamped.pose
        
        diameter = radius * 2.0
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.5

        self.viz_sphere_pub.publish(marker)
    
    def publish_arrow_marker(self, pose_stamped):
        marker = Marker()
        marker.header = pose_stamped.header
        marker.ns = "target_arrow"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        marker.pose = pose_stamped.pose
        
        marker.scale.x = 0.15  # 15cm Long
        marker.scale.y = 0.01  # 1cm Wide
        marker.scale.z = 0.02  # 2cm Head
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        self.viz_pub.publish(marker)

    def filter_objects_above_table(self, object_cloud, plane_model):
        """
        Removes points that are on the wrong side of the table.
        Also removes points that are too high to be relevant objects.
        """
        if len(object_cloud.points) == 0:
            return object_cloud

        a, b, c, d = plane_model

        # Determine camera side sign
        camera_sign = np.sign(d)

        points = np.asarray(object_cloud.points)
        colors = np.asarray(object_cloud.colors) if object_cloud.has_colors() else None
        
        # Calculate signed distance for all points
        distances = (points[:,0] * a) + (points[:,1] * b) + (points[:,2] * c) + d
        
        # Filter
        # - Must be on the same side as camera (Above table)
        # - Must be within 50cm of the table surface (Ignore ceiling/high noise)
        valid_mask = (np.sign(distances) == camera_sign) & (np.abs(distances) < 0.5)

        filtered_cloud = o3d.geometry.PointCloud()
        filtered_cloud.points = o3d.utility.Vector3dVector(points[valid_mask])
        if colors is not None:
            filtered_cloud.colors = o3d.utility.Vector3dVector(colors[valid_mask])
            
        return filtered_cloud
    
    def get_orientation_from_plane(self, plane_model):
        """
        Calculates a quaternion where the X-AXIS points into the table.
        """
        # Get the Normal Vector [a,b,c]
        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)

        # Define the arrow direction (X-axis)
        if normal[1] < 0:  # If Y is negative (pointing up)
            target_x = -normal # Flip it to point down/table
        else:
            target_x = normal

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
    
    def find_empty_spot(self, table_cloud, object_cloud, radius_to_check, plane_model):
        if len(table_cloud.points) == 0: return None
        
        a, b, c, d = plane_model

        # Get table geometry statistics
        table_pts = np.asarray(table_cloud.points)
        min_x, min_y = np.min(table_pts[:,0]), np.min(table_pts[:,1])
        max_x, max_y = np.max(table_pts[:,0]), np.max(table_pts[:,1])

        avg_z = np.mean(table_pts[:,2]) # The height of the table plane
        center_x, center_y = np.mean(table_pts[:,0]), np.mean(table_pts[:,1])

        table_tree = o3d.geometry.KDTreeFlann(table_cloud)
        
        has_objects = len(object_cloud.points) > 0
        object_tree = None

        if has_objects:
            # Create a shadow cloud where all object points are projected onto the table plane (Z = avg_z).
            # This ensures that tall objects block the grid points underneath them.
            obj_points = np.asarray(object_cloud.points).copy()
            obj_points[:, 2] = avg_z  # Force Z to match table height
            
            flat_object_cloud = o3d.geometry.PointCloud()
            flat_object_cloud.points = o3d.utility.Vector3dVector(obj_points)
            
            # Build tree on the flattened cloud
            object_tree = o3d.geometry.KDTreeFlann(flat_object_cloud)

        # Grid search
        step = 0.05
        best_pt = None
        best_score = -float('inf')

        for x in np.arange(min_x, max_x, step):
            for y in np.arange(min_y , max_y, step):

                if abs(c) < 0.001: # Avoid division by zero (vertical plane)
                    z_plane = avg_z
                else:
                    z_plane = -(a*x + b*y + d) / c
                
                # The candidate point follows the table tilt
                cand = np.array([x, y, z_plane])
                
                # use a temp candidate at avg_z for the KDTree checks
                # (the tree and objects are built around avg_z for simplicity)
                check_cand = np.array([x, y, avg_z])
                
                # Check if it is this actually on the table
                [k, _, _] = table_tree.search_radius_vector_3d(check_cand, 0.05)
                if k == 0: continue 

                dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Check collision with flattened objects
                if has_objects:
                    # Finds distance to the nearest object shadow
                    [_, _, d_sq] = object_tree.search_knn_vector_3d(check_cand, 1)
                    dist_obj = np.sqrt(d_sq[0])
                    
                    # Radius Check
                    if dist_obj < radius_to_check: 
                        continue
                    
                    # Score: Maximize distance to object, minimize distance to center
                    score = dist_obj - (0.8 * dist_center)
                else:
                    score = -dist_center

                # Check if object radius fits on table
                if (x - radius_to_check < min_x) or (x + radius_to_check > max_x):
                    continue
                if (y - radius_to_check < min_y) or (y + radius_to_check > max_y):
                    continue

                if score > best_score:
                    best_score = score
                    best_pt = cand
                    
        return best_pt

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
