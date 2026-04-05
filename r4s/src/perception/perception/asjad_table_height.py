import rclpy
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from std_msgs.msg import Float32
import tf2_ros
from scipy.spatial.transform import Rotation

WORLD_FRAME = 'eddie_base_footprint'
TABLE_Z_MIN = 0.15   # m — minimum plausible table height above floor
TABLE_Z_MAX = 1.50   # m — maximum plausible table height above floor


class TableHeightNode(LifecycleNode):
    def __init__(self):
        super().__init__('table_height_estimator')

        self.declare_parameter('input_mode', 'realsense')
        self.declare_parameter('max_distance_m', 2.5)

        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.max_distance = float(
            self.get_parameter('max_distance_m').get_parameter_value().double_value
        )

        if input_mode == 'robot':
            self.pc_topic = '/camera/depth/color/points'
        else:
            self.pc_topic = '/camera/camera/depth/color/points'

        self.marker_pub  = None
        self.debug_pc_pub = None
        self.height_pub  = None
        self.pc_sub      = None
        self.tf_buffer   = None
        self.tf_listener = None
        self._is_active  = False

        self.get_logger().info(
            f"Table Height Node initialised — topic: {self.pc_topic}"
        )

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring…")
        try:
            self.marker_pub   = self.create_lifecycle_publisher(
                MarkerArray, '/table_height_visualization', 10)
            self.debug_pc_pub = self.create_lifecycle_publisher(
                PointCloud2,  '/table_points_debug', 10)
            self.height_pub   = self.create_lifecycle_publisher(
                Float32, '/table_height_value', 10)
            self.pc_sub = self.create_subscription(
                PointCloud2, self.pc_topic, self._callback, 10)

            self.tf_buffer   = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Configuration failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        super().on_activate(state)
        self._is_active = True
        self.get_logger().info("Activated — processing point clouds.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self._is_active = False
        super().on_deactivate(state)
        self.get_logger().info("Deactivated.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up…")
        for pub in (self.marker_pub, self.debug_pc_pub, self.height_pub):
            if pub is not None:
                self.destroy_publisher(pub)
        if self.pc_sub is not None:
            self.destroy_subscription(self.pc_sub)
        self.marker_pub = self.debug_pc_pub = self.height_pub = None
        self.pc_sub     = None
        self.tf_buffer  = None
        self.tf_listener = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------ #
    #  Processing                                                          #
    # ------------------------------------------------------------------ #

    def _callback(self, pc_msg: PointCloud2):
        if not self._is_active:
            return

        # 1. Read points
        xyz = pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        if xyz.size == 0 or xyz.ndim != 2 or xyz.shape[1] < 3:
            return
        xyz = xyz[np.linalg.norm(xyz, axis=1) < self.max_distance]
        if len(xyz) < 500:
            return

        # 2. Transform to world frame — skip frame entirely if TF unavailable.
        #    Try exact stamp first (correct for both live robot and bag replay).
        #    Fall back to latest available only if the exact stamp is not yet
        #    in the buffer (handles minor clock skew on the real robot).
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                WORLD_FRAME,
                pc_msg.header.frame_id,
                pc_msg.header.stamp,
                timeout=Duration(seconds=0.2),
            )
        except Exception:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    WORLD_FRAME,
                    pc_msg.header.frame_id,
                    Time(),
                )
            except Exception as e:
                self.get_logger().warn(
                    f"TF unavailable ({e}) — skipping frame",
                    throttle_duration_sec=2.0,
                )
                return

        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        world_pts = (
            Rotation.from_quat([q.x, q.y, q.z, q.w]).apply(xyz)
            + np.array([t.x, t.y, t.z])
        )

        # 3. Restrict to plausible table-height band.
        #    In eddie_base_footprint Z=0 is the floor — no floor detection needed.
        band = (world_pts[:, 2] > TABLE_Z_MIN) & (world_pts[:, 2] < TABLE_Z_MAX)
        ransac_pts = world_pts[band]
        if len(ransac_pts) < 300:
            self.get_logger().debug("Too few points in table-height band.")
            return

        # 4. Downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ransac_pts.astype(np.float64))
        pcd_work = pcd.voxel_down_sample(voxel_size=0.015)

        # 5. Iterative RANSAC — strip non-horizontal planes until we find
        #    the table surface (or exhaust retries).
        up = np.array([0.0, 0.0, 1.0])
        table_cloud = None
        for iteration in range(5):
            if len(pcd_work.points) < 100:
                break
            try:
                plane_model, inliers = pcd_work.segment_plane(
                    distance_threshold=0.015, ransac_n=3, num_iterations=1000
                )
            except Exception:
                break
            if len(inliers) < 100:
                break

            normal = np.array(plane_model[:3])
            normal /= np.linalg.norm(normal)
            angle_deg = np.rad2deg(
                np.arccos(np.clip(np.abs(normal.dot(up)), -1.0, 1.0))
            )

            if angle_deg <= 15.0:
                table_cloud = pcd_work.select_by_index(inliers)
                break

            self.get_logger().debug(
                f"Iter {iteration}: non-horizontal plane ({angle_deg:.1f}°), stripping."
            )
            pcd_work = pcd_work.select_by_index(inliers, invert=True)

        if table_cloud is None:
            self.get_logger().debug("No horizontal plane found — skipping frame.")
            return

        # 6. Compute height = median Z of table surface
        table_pts = np.asarray(table_cloud.points)
        height_m  = float(np.median(table_pts[:, 2]))

        if not (TABLE_Z_MIN <= height_m <= TABLE_Z_MAX):
            self.get_logger().debug(f"Height {height_m:.3f} m outside valid band — skipping.")
            return

        t_x = float(np.median(table_pts[:, 0]))
        t_y = float(np.median(table_pts[:, 1]))

        self.get_logger().info(f"Table height: {height_m:.3f} m (world frame)")

        # 7. Publish
        import copy
        hdr = copy.copy(pc_msg.header)
        hdr.frame_id = WORLD_FRAME

        self._publish_markers(t_x, t_y, height_m, hdr)

        self.debug_pc_pub.publish(
            pc2.create_cloud_xyz32(hdr, table_pts.astype(np.float32))
        )

    def _publish_markers(self, tx: float, ty: float, height_m: float, header):
        ma = MarkerArray()

        # Red sphere at table surface
        sphere = Marker()
        sphere.header = header
        sphere.ns, sphere.id = "table", 0
        sphere.type   = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = tx
        sphere.pose.position.y = ty
        sphere.pose.position.z = height_m
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.08
        sphere.color.r, sphere.color.a = 1.0, 1.0
        ma.markers.append(sphere)

        # Green disc at floor (Z = 0)
        disc = Marker()
        disc.header = header
        disc.ns, disc.id = "floor", 1
        disc.type   = Marker.CYLINDER
        disc.action = Marker.ADD
        disc.pose.position.x = tx
        disc.pose.position.y = ty
        disc.pose.position.z = 0.0
        disc.scale.x = disc.scale.y = 0.3
        disc.scale.z = 0.005
        disc.color.g, disc.color.a = 1.0, 1.0
        ma.markers.append(disc)

        # Yellow vertical line floor → table
        line = Marker()
        line.header = header
        line.ns, line.id = "line", 2
        line.type   = Marker.LINE_LIST
        line.action = Marker.ADD
        line.scale.x = 0.005
        line.color.r = line.color.g = line.color.a = 1.0
        line.points.append(disc.pose.position)
        line.points.append(sphere.pose.position)
        ma.markers.append(line)

        # White text label at mid-height
        text = Marker()
        text.header = header
        text.ns, text.id = "text", 3
        text.type   = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.text   = f"{height_m:.2f} m"
        text.pose.position.x = tx + 0.15
        text.pose.position.y = ty
        text.pose.position.z = height_m / 2.0
        text.scale.z = 0.06
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        ma.markers.append(text)

        self.marker_pub.publish(ma)

        msg = Float32()
        msg.data = height_m
        self.height_pub.publish(msg)


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
