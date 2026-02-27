import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2

# Import QoS classes
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# --- FIX 1: Import traceback for correct error logging ---
import traceback

class FloorDetectorNode(Node):
    def __init__(self):
        super().__init__('floor_detector_node')
        
        # --- Parameters ---
        self.declare_parameter('input_topic', '/camera/depth/color/points')
        self.declare_parameter('distance_threshold', 0.02) # 2cm
        self.declare_parameter('angle_threshold_deg', 45.0) # 10 degrees from Z-axis
        
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.distance_threshold = self.get_parameter('distance_threshold').get_parameter_value().double_value
        self.angle_threshold_rad = np.deg2rad(self.get_parameter('angle_threshold_deg').get_parameter_value().double_value)

        # --- Define the Sensor QoS Profile ---
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Subscriber ---
        self.subscription = self.create_subscription(
            PointCloud2,
            input_topic,
            self.cloud_callback,
            qos_profile  # Use the sensor-specific QoS
        )
        
        # --- Publishers ---
        self.floor_pub = self.create_publisher(PointCloud2, '/floor_cloud', 10)
        self.other_pub = self.create_publisher(PointCloud2, '/other_cloud', 10)
        
        self.get_logger().info(f"Floor Detector Node started. Subscribing to '{input_topic}'")

    def cloud_callback(self, msg):
        """Main callback: receives a point cloud, finds the floor, and republishes."""
        self.get_logger().info("Callback triggered! Processing cloud...")
        
        try:
            # --- FIX 2: Simplify the conversion ---
            # We assume read_points_numpy with field_names gives us a simple (N, 3) array
            xyz_data = pc2.read_points_numpy(
                msg, 
                field_names=("x", "y", "z"), 
                skip_nans=True
            )
            
            if xyz_data.size == 0:
                self.get_logger().warn("Point list is EMPTY. Is the camera covered or facing the sky?")
                return

            self.get_logger().info(f"Successfully converted to array with shape: {xyz_data.shape}")

        except Exception as e:
            # --- FIX 1: Correct error logging ---
            self.get_logger().error(f"Failed to convert PointCloud2 to NumPy: {e}")
            self.get_logger().error(f"TRACEBACK: {traceback.format_exc()}")
            return
            # --- End Fix 1 ---

        if xyz_data.size == 0:
            self.get_logger().info("Numpy array is empty. Skipping.")
            return

        # 3. Convert NumPy array to Open3D PointCloud object
        pcd = o3d.geometry.PointCloud()
        
        # Open3D's Vector3dVector expects float64, so we ensure the type is correct.
        pcd.points = o3d.utility.Vector3dVector(xyz_data.astype(np.float64))

        # 4. Run RANSAC to find the largest plane
        self.get_logger().info("Running RANSAC...")
        try:
            plane_model, inlier_indices = pcd.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
        except Exception as e:
            self.get_logger().warn(f"RANSAC segmentation failed: {e}")
            self.get_logger().warn(f"TRACEBACK: {traceback.format_exc()}")
            return
            
        if not inlier_indices:
            self.get_logger().info("No plane found. Publishing all points as 'other'.")
            self.other_pub.publish(msg) # Publish original cloud
            return

        # 5. Check if the plane is the "floor"
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal) # Normalize
        
        ### --- THIS IS THE FIX --- ###
        # "Up" in the camera frame is the Y-axis, not the Z-axis.
        up_vector = np.array([0, 1, 0])
        
        angle_rad = np.arccos(np.clip(np.abs(normal.dot(up_vector)), -1.0, 1.0))
        ### --- END OF FIX --- ###
        
        angle_deg = np.rad2deg(angle_rad)
        
        # 6. Select points based on whether the floor was found
        if angle_rad <= self.angle_threshold_rad:
            # (rest of your code is correct)
            self.get_logger().info(f"Floor plane found with {len(inlier_indices)} points. Angle: {angle_deg:.1f} deg")
            floor_pcd = pcd.select_by_index(inlier_indices)
            other_pcd = pcd.select_by_index(inlier_indices, invert=True)
            floor_points = np.asarray(floor_pcd.points)
            other_points = np.asarray(other_pcd.points)
        else:
            self.get_logger().info(f"Plane found (angle: {angle_deg:.1f} deg), but it's not the floor.")
            floor_points = np.array([]) # Empty array
            other_points = xyz_data # All points are "other"

        # 7. Convert NumPy arrays back to ROS PointCloud2 messages
        header = msg.header # Re-use the original header
        
        if floor_points.size > 0:
            floor_msg = self.create_point_cloud_msg(floor_points, header)
            self.floor_pub.publish(floor_msg)
            
        if other_points.size > 0:
            other_msg = self.create_point_cloud_msg(other_points, header)
            self.other_pub.publish(other_msg)

    def create_point_cloud_msg(self, points, header: Header) -> PointCloud2:
        """Helper function to create a PointCloud2 message from a NumPy array."""
        # This function creates a PointCloud2 message for only XYZ data
        # `points` is an (N, 3) NumPy array
        return pc2.create_cloud_xyz32(header, points.astype(np.float32))


def main(args=None):
    rclpy.init(args=args)
    node = FloorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()