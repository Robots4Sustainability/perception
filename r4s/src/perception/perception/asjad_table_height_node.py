import rclpy
import rclpy.duration
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from std_msgs.msg import Float32
import tf2_ros
from scipy.spatial.transform import Rotation

# Robot world frame — Z points up, gravity-aligned
WORLD_FRAME = 'eddie_base_footprint'


class TableHeightNode(LifecycleNode):
    def __init__(self):
        super().__init__('table_height_estimator')

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

        self.det_model = None
        self.marker_pub = None
        self.debug_pc_pub = None
        self.height_pub = None
        self.pc_sub = None
        self.tf_buffer = None
        self.tf_listener = None
        self._is_active = False
        self.get_logger().info("Table Height Lifecycle Node Initialized (Unconfigured).")

    # --- LIFECYCLE CALLBACKS ---

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring: lightweight geometric mode (no heavy models)")
        try:
            self.marker_pub = self.create_lifecycle_publisher(MarkerArray, '/table_height_visualization', 10)
            self.debug_pc_pub = self.create_lifecycle_publisher(PointCloud2, '/table_points_debug', 10)
            self.height_pub = self.create_lifecycle_publisher(Float32, '/table_height_value', 10)
            self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.callback, 10)

            # TF buffer + listener for world-frame transform
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            self.get_logger().info("Configuration complete.")
            self.get_logger().info("returned 1")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating node...")
        super().on_activate(state)
        self._is_active = True
        self.get_logger().info("Node activated. Processing incoming data.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating node...")
        self._is_active = False
        super().on_deactivate(state)
        self.get_logger().info("Node deactivated. Pausing processing.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up resources...")
        self.destroy_publisher(self.marker_pub)
        self.destroy_publisher(self.debug_pc_pub)
        self.destroy_publisher(self.height_pub)
        if self.pc_sub is not None:
            self.destroy_subscription(self.pc_sub)
        self.marker_pub = self.debug_pc_pub = self.height_pub = None
        self.pc_sub = None
        self.tf_buffer = None
        self.tf_listener = None
        self.get_logger().info("Cleanup complete.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down node...")
        if self.det_model is not None:
            self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # --- HELPERS ---

    def _apply_transform(self, points, transform_stamped):
        """Apply a TF TransformStamped to an Nx3 numpy array of points."""
        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation
        translation = np.array([t.x, t.y, t.z])
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        return rot.apply(points) + translation

    # --- PROCESSING CALLBACK ---

    def callback(self, pc_msg):
        if not self._is_active:
            return
        self.process_point_cloud(pc_msg)

    def process_point_cloud(self, pc_msg):
        if pc_msg.height * pc_msg.width == 0:
            return

        xyz_data = pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        if xyz_data.size == 0 or len(xyz_data.shape) != 2 or xyz_data.shape[1] < 3:
            return

        valid_distances = np.linalg.norm(xyz_data, axis=1) < self.max_distance
        points = xyz_data[valid_distances]
        if len(points) < 500:
            return

        # --- Try to transform points into the world frame via TF ---
        use_world_frame = False
        work_points = points
        marker_frame = pc_msg.header.frame_id

        if self.tf_buffer is not None:
            try:
                transform = self.tf_buffer.lookup_transform(
                    WORLD_FRAME,
                    pc_msg.header.frame_id,
                    pc_msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                work_points = self._apply_transform(points, transform)
                use_world_frame = True
                marker_frame = WORLD_FRAME
            except Exception:
                pass  # fall back to camera-frame processing

        if use_world_frame:
            self.get_logger().debug("Using world frame (TF available)")
        else:
            self.get_logger().debug("Using camera frame (no TF)")

        # up_idx: axis index for "vertical" direction
        # world frame: Z (index 2) is up
        # camera frame: Y (index 1) is up (approximate)
        up_idx = 2 if use_world_frame else 1
        up_vector = np.array([0, 0, 1]) if use_world_frame else np.array([0, 1, 0])

        # --- In world frame: restrict RANSAC to table-height points only ---
        # This prevents RANSAC picking walls or robot body as dominant plane
        if use_world_frame:
            floor_approx = np.percentile(work_points[:, 2], 5)
            table_mask = (work_points[:, 2] > floor_approx + 0.2) & \
                         (work_points[:, 2] < floor_approx + 1.2)
            ransac_pts = work_points[table_mask]
            if len(ransac_pts) < 300:
                ransac_pts = work_points  # fall back to all points
        else:
            ransac_pts = work_points

        # --- Downsample and find dominant plane (table surface) ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ransac_pts.astype(np.float64))
        pcd_down = pcd.voxel_down_sample(voxel_size=0.015)

        try:
            plane_model, inliers = pcd_down.segment_plane(
                distance_threshold=0.015, ransac_n=3, num_iterations=1000
            )
        except Exception:
            return

        if len(inliers) < 100:
            return

        # Validate plane is roughly horizontal
        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)
        angle_rad = np.arccos(np.clip(np.abs(normal.dot(up_vector)), -1.0, 1.0))
        if np.rad2deg(angle_rad) > 85.0:
            self.get_logger().info(f"returned 9, angle was {np.rad2deg(angle_rad):.1f} deg")
            return

        table_cloud = pcd_down.select_by_index(inliers)
        table_pts = np.asarray(table_cloud.points)

        # Table centre
        t_x = np.median(table_pts[:, 0])
        t_y = np.median(table_pts[:, 1])
        t_z = np.median(table_pts[:, 2])
        t_up = np.median(table_pts[:, up_idx])   # "height" of table surface

        # --- Floor estimation ---
        if use_world_frame:
            # In eddie_base_footprint, Z=0 is the ground plane by definition.
            # Table height is simply the Z coordinate of the table surface.
            floor_up_at_table = 0.0
            height_m = t_up
        else:
            # Camera frame: estimate floor via RANSAC on non-table points
            pcd_full = o3d.geometry.PointCloud()
            pcd_full.points = o3d.utility.Vector3dVector(work_points.astype(np.float64))
            pcd_full_down = pcd_full.voxel_down_sample(voxel_size=0.02)
            all_pts = np.asarray(pcd_full_down.points)
            table_plane_d = plane_model[3]
            distances = np.abs(all_pts @ np.array(plane_model[:3]) + table_plane_d)
            other_pts = all_pts[distances > 0.03]

            up_all = work_points[:, up_idx]
            floor_seed = np.percentile(up_all, 90) if normal[up_idx] < 0 else np.percentile(up_all, 10)
            floor_mask = (other_pts[:, up_idx] > t_up + 0.1) if normal[up_idx] < 0 \
                         else (other_pts[:, up_idx] < t_up - 0.1)

            floor_candidates = other_pts[floor_mask]
            floor_up_at_table = None

            if len(floor_candidates) > 50:
                floor_candidates = floor_candidates[::max(1, len(floor_candidates) // 800)]
                X_in = floor_candidates[:, [0, 2]]   # X, Z → predict Y
                Y_out = floor_candidates[:, 1]
                pred_input = [[t_x, t_z]]
                ransac = RANSACRegressor(residual_threshold=0.05)
                try:
                    ransac.fit(X_in, Y_out)
                    floor_up_at_table = ransac.predict(pred_input)[0]
                except Exception:
                    floor_up_at_table = None

            if floor_up_at_table is None:
                floor_up_at_table = floor_seed

            height_m = abs(t_up - floor_up_at_table)
        if height_m < 0.15 or height_m > 1.5:
            self.get_logger().info(f"returned 10, height was {height_m:.3f}m")
            return

        self.get_logger().info(f"Table height: {height_m:.3f}m ({'world' if use_world_frame else 'camera'} frame)")

        # Build marker header with correct frame
        from copy import deepcopy
        marker_header = deepcopy(pc_msg.header)
        marker_header.frame_id = marker_frame

        self._publish_markers(t_x, t_y, t_z, t_up, floor_up_at_table, height_m,
                              marker_header, use_world_frame)

        debug_cloud = pc2.create_cloud_xyz32(marker_header, table_pts.astype(np.float32))
        self.debug_pc_pub.publish(debug_cloud)

    def _publish_markers(self, tx, ty, tz, t_up, floor_up, height_m,
                         header, use_world_frame):
        marker_array = MarkerArray()

        # --- Table sphere (red) ---
        m_table = Marker()
        m_table.header = header
        m_table.ns, m_table.id = "table", 0
        m_table.type = Marker.SPHERE
        m_table.action = Marker.ADD
        if use_world_frame:
            m_table.pose.position.x = float(tx)
            m_table.pose.position.y = float(ty)
            m_table.pose.position.z = float(t_up)
        else:
            m_table.pose.position.x = float(tx)
            m_table.pose.position.y = float(t_up)
            m_table.pose.position.z = float(tz)
        m_table.scale.x = m_table.scale.y = m_table.scale.z = 0.08
        m_table.color.r, m_table.color.g, m_table.color.b, m_table.color.a = 1.0, 0.0, 0.0, 1.0
        marker_array.markers.append(m_table)

        # --- Floor cube (green) ---
        m_floor = Marker()
        m_floor.header = header
        m_floor.ns, m_floor.id = "floor", 1
        m_floor.type = Marker.CUBE
        m_floor.action = Marker.ADD
        if use_world_frame:
            m_floor.pose.position.x = float(tx)
            m_floor.pose.position.y = float(ty)
            m_floor.pose.position.z = float(floor_up)
            m_floor.scale.x, m_floor.scale.y, m_floor.scale.z = 0.3, 0.3, 0.005
        else:
            m_floor.pose.position.x = float(tx)
            m_floor.pose.position.y = float(floor_up)
            m_floor.pose.position.z = float(tz)
            m_floor.scale.x, m_floor.scale.z = 0.2, 0.2
            m_floor.scale.y = 0.005
        m_floor.color.r, m_floor.color.g, m_floor.color.b, m_floor.color.a = 0.0, 1.0, 0.0, 1.0
        marker_array.markers.append(m_floor)

        # --- Line (yellow) ---
        m_line = Marker()
        m_line.header = header
        m_line.ns, m_line.id = "line", 2
        m_line.type = Marker.LINE_LIST
        m_line.action = Marker.ADD
        m_line.scale.x = 0.005
        m_line.color.r, m_line.color.g, m_line.color.b, m_line.color.a = 1.0, 1.0, 0.0, 1.0
        m_line.points.append(m_table.pose.position)
        m_line.points.append(m_floor.pose.position)
        marker_array.markers.append(m_line)

        # --- Text label (white) ---
        m_text = Marker()
        m_text.header = header
        m_text.ns, m_text.id = "text", 3
        m_text.type = Marker.TEXT_VIEW_FACING
        m_text.action = Marker.ADD
        m_text.text = f"{height_m:.2f}m"
        if use_world_frame:
            m_text.pose.position.x = float(tx) + 0.15
            m_text.pose.position.y = float(ty)
            m_text.pose.position.z = (t_up + floor_up) / 2.0
        else:
            m_text.pose.position.x = float(tx) + 0.15
            m_text.pose.position.y = (t_up + floor_up) / 2.0
            m_text.pose.position.z = float(tz)
        m_text.scale.z = 0.05
        m_text.color.r, m_text.color.g, m_text.color.b, m_text.color.a = 1.0, 1.0, 1.0, 1.0
        marker_array.markers.append(m_text)

        # Publish height value
        height_msg = Float32()
        height_msg.data = float(height_m)
        if self.height_pub is not None:
            self.height_pub.publish(height_msg)

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
