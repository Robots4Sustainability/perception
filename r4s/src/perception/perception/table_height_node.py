import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from std_msgs.msg import Float32

class TableHeightNode(LifecycleNode):
    def __init__(self):
        super().__init__('table_height_estimator')

        # --- PARAMETERS ---
        self.declare_parameter('input_mode', 'realsense')
        self.declare_parameter('max_distance_m', 2.5)
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.max_distance = float(self.get_parameter('max_distance_m').get_parameter_value().double_value)

        if input_mode == 'robot':
            self.pc_topic = '/camera/depth/color/points'
        elif input_mode == 'realsense':
            self.pc_topic = '/camera/camera/depth/color/points'
        else:
            self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'realsense'")
            self.pc_topic = '/camera/camera/depth/color/points'

        # Placeholders for resources
        self.det_model = None  # kept for lifecycle shutdown logic
        self.marker_pub = None
        self.debug_pc_pub = None
        self.height_pub = None
        
        self.pc_sub = None

        self._is_active = False
        self.get_logger().info("Table Height Lifecycle Node Initialized (Unconfigured).")

    # --- LIFECYCLE CALLBACKS ---

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring: lightweight geometric mode (no heavy models)")
        try:
            # Create Lifecycle Publishers
            self.marker_pub = self.create_lifecycle_publisher(MarkerArray, '/table_height_visualization', 10)
            self.debug_pc_pub = self.create_lifecycle_publisher(PointCloud2, '/table_points_debug', 10)
            self.height_pub = self.create_lifecycle_publisher(Float32, '/table_height_value', 10)

            # Create subscription
            self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.callback, 10)

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating node...")
        # super().on_activate() is required to activate the lifecycle publishers
        super().on_activate(state)
        self._is_active = True
        self.get_logger().info("Node activated. Processing incoming data.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating node...")
        self._is_active = False
        # super().on_deactivate() deactivates the lifecycle publishers
        super().on_deactivate(state)
        self.get_logger().info("Node deactivated. Pausing processing.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up resources...")
        
        # 1. Destroy Subscriptions and Publishers
        self.destroy_publisher(self.marker_pub)
        self.destroy_publisher(self.debug_pc_pub)
        self.destroy_publisher(self.height_pub)
        
        if self.pc_sub is not None:
            self.destroy_subscription(self.pc_sub)

        self.marker_pub = self.debug_pc_pub = None
        self.pc_sub = None

        self.get_logger().info("Cleanup complete.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down node...")
        # If shutting down from active/configured, ensure cleanup happens
        if self.det_model is not None:
            self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # --- PROCESSING CALLBACK ---

    def callback(self, pc_msg):
        # Guard clause: Do nothing if the node is not active
        if not self._is_active:
            return

        self.process_point_cloud(pc_msg)
    def process_point_cloud(self, pc_msg):
        if pc_msg.height * pc_msg.width == 0:
            return

        xyz_data = pc2.read_points_numpy(
            pc_msg,
            field_names=("x", "y", "z"),
            skip_nans=True
        )

        if xyz_data.size == 0 or len(xyz_data.shape) != 2 or xyz_data.shape[1] < 3:
            return

        valid_distances = np.linalg.norm(xyz_data, axis=1) < self.max_distance
        points = xyz_data[valid_distances]
        if len(points) < 500:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd_down = pcd.voxel_down_sample(voxel_size=0.015)

        try:
            plane_model, inliers = pcd_down.segment_plane(
                distance_threshold=0.015,
                ransac_n=3,
                num_iterations=1000
            )
        except Exception:
            return

        if len(inliers) < 100:
            return

        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)
        up_vector = np.array([0, 1, 0])
        angle_rad = np.arccos(np.clip(np.abs(normal.dot(up_vector)), -1.0, 1.0))
        if np.rad2deg(angle_rad) > 30.0:
            return

        table_cloud_o3d = pcd_down.select_by_index(inliers)
        other_cloud_o3d = pcd_down.select_by_index(inliers, invert=True)
        table_pts = np.asarray(table_cloud_o3d.points)
        other_pts = np.asarray(other_cloud_o3d.points)

        t_x = np.median(table_pts[:, 0])
        t_y = np.median(table_pts[:, 1])
        t_z = np.median(table_pts[:, 2])

        y_all = points[:, 1]
        floor_seed = np.percentile(y_all, 90) if normal[1] < 0 else np.percentile(y_all, 10)

        floor_candidates_mask = (other_pts[:, 1] > t_y + 0.1) if normal[1] < 0 else (other_pts[:, 1] < t_y - 0.1)
        floor_candidates = other_pts[floor_candidates_mask]
        floor_y_at_table = None

        if len(floor_candidates) > 50:
            floor_candidates = floor_candidates[::max(1, len(floor_candidates) // 800)]
            X_in = floor_candidates[:, [0, 2]]
            Y_out = floor_candidates[:, 1]
            ransac = RANSACRegressor(residual_threshold=0.05)
            try:
                ransac.fit(X_in, Y_out)
                floor_y_at_table = ransac.predict([[t_x, t_z]])[0]
            except Exception:
                floor_y_at_table = None

        if floor_y_at_table is None:
            floor_y_at_table = floor_seed

        height_m = abs(floor_y_at_table - t_y)
        if height_m < 0.15 or height_m > 1.5:
            return

        self.publish_markers(t_x, t_y, t_z, floor_y_at_table, pc_msg.header)
        debug_cloud = pc2.create_cloud_xyz32(pc_msg.header, table_pts.astype(np.float32))
        self.debug_pc_pub.publish(debug_cloud)

    def publish_markers(self, tx, ty, tz, fy, header):
        marker_array = MarkerArray()

        m_table = Marker()
        m_table.header = header
        m_table.ns = "table"
        m_table.id = 0
        m_table.type = Marker.SPHERE
        m_table.action = Marker.ADD
        m_table.pose.position.x, m_table.pose.position.y, m_table.pose.position.z = float(tx), float(ty), float(tz)
        m_table.scale.x = m_table.scale.y = m_table.scale.z = 0.08
        m_table.color.r, m_table.color.g, m_table.color.b, m_table.color.a = 1.0, 0.0, 0.0, 1.0
        marker_array.markers.append(m_table)

        log_msg = f"Table Z: {tz:.2f}m"

        if fy is not None:
            height_meters = abs(fy - ty)
            log_msg += f" | Floor Est: {fy:.2f}m | HEIGHT: {height_meters:.3f}m"

            height_msg = Float32()
            height_msg.data = float(height_meters)
            if self.height_pub is not None:
                self.height_pub.publish(height_msg)

            m_floor = Marker()
            m_floor.header = header
            m_floor.ns = "floor"
            m_floor.id = 1
            m_floor.type = Marker.CUBE
            m_floor.action = Marker.ADD
            m_floor.pose.position.x, m_floor.pose.position.y, m_floor.pose.position.z = float(tx), float(fy), float(tz)
            m_floor.scale.x, m_floor.scale.z = 0.2, 0.2
            m_floor.scale.y = 0.005 
            m_floor.color.r, m_floor.color.g, m_floor.color.b, m_floor.color.a = 0.0, 1.0, 0.0, 1.0
            marker_array.markers.append(m_floor)

            m_line = Marker()
            m_line.header = header
            m_line.ns = "line"
            m_line.id = 2
            m_line.type = Marker.LINE_LIST
            m_line.action = Marker.ADD
            m_line.scale.x = 0.005 
            m_line.color.r, m_line.color.g, m_line.color.b, m_line.color.a = 1.0, 1.0, 0.0, 1.0
            m_line.points.append(m_table.pose.position)
            m_line.points.append(m_floor.pose.position)
            marker_array.markers.append(m_line)

            m_text = Marker()
            m_text.header = header
            m_text.ns = "text"
            m_text.id = 3
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.text = f"{height_meters:.2f}m"
            m_text.pose.position.x = float(tx) + 0.15
            m_text.pose.position.y = (ty + fy) / 2.0  
            m_text.pose.position.z = float(tz)
            m_text.scale.z = 0.05 
            m_text.color.r, m_text.color.g, m_text.color.b, m_text.color.a = 1.0, 1.0, 1.0, 1.0
            marker_array.markers.append(m_text)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args) 
    node = TableHeightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
