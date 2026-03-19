import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Header
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformException, TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct

class TableHeightLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('table_height_estimator')

        self.declare_parameter('input_mode', 'robot')
        self.declare_parameter('base_frame', 'eddie_base_footprint') 
        
        self.input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        if self.input_mode == 'robot':
            self.pc_topic = '/camera/depth/color/points'
        else:
            self.pc_topic = '/camera/camera/depth/color/points'

        # Permanent Resources (TF2)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self._is_active = False
        
        # Placeholders for Publishers/Subscribers
        self.pc_sub = None
        self.height_pub = None
        self.pose_pub = None
        self.table_cloud_pub = None
        self.viz_pub = None

        self.get_logger().info("Table Height Lifecycle Node Initialized (Unconfigured).")

    # Lifecycle Callbacks

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring Table Height Estimator...")
        
        try:
            # Initialize Publishers
            self.height_pub = self.create_lifecycle_publisher(Float32, '/table_height_value', 10)
            self.pose_pub = self.create_lifecycle_publisher(PoseStamped, '/perception/table_pose', 10)
            self.table_cloud_pub = self.create_lifecycle_publisher(PointCloud2, '/perception/debug/table_height_plane', 10)
            self.viz_pub = self.create_lifecycle_publisher(MarkerArray, '/perception/debug/table_height_viz', 10)

            # Initialize Subscriber
            self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.cloud_callback, 10)

            self.get_logger().info("Configuration Complete.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Configuration Failed: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating Table Height Estimator...")
        self._is_active = True
        return super().on_activate(state)

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating Table Height Estimator...")
        self._is_active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up Table Height Estimator...")
        self.destroy_subscription(self.pc_sub)
        return TransitionCallbackReturn.SUCCESS

    # Cloud Processing Callback

    def cloud_callback(self, ros_cloud_msg):
        if not self._is_active:
            return

        # Get Transform
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.base_frame, ros_cloud_msg.header.frame_id, rclpy.time.Time()
            )
        except TransformException:
            return

        # Convert and Transform Cloud
        pcd = self.convert_ros_to_o3d(ros_cloud_msg)
        if pcd is None or len(pcd.points) == 0: return

        tf_matrix = self.transform_to_matrix(transform_stamped)
        pcd.transform(tf_matrix)
        pcd_down = pcd.voxel_down_sample(voxel_size=0.01)

        # Iterative RANSAC 
        # Wall and floor Rejection)
        table_cloud, table_height, center_x, center_y, final_plane_model = None, 0.0, 0.0, 0.0, None
        
        for _ in range(5):
            try:
                plane_model, inliers = pcd_down.segment_plane(0.015, 3, 1000)
            except: break 

            temp_cloud = pcd_down.select_by_index(inliers)
            temp_pts = np.asarray(temp_cloud.points)
            is_horizontal = abs(plane_model[2]) > 0.85 
            avg_z = np.mean(temp_pts[:, 2])

            if not is_horizontal or avg_z < 0.2:
                pcd_down = pcd_down.select_by_index(inliers, invert=True)
                continue
            else:
                table_cloud, table_height = temp_cloud, float(avg_z)
                center_x, center_y = float(np.mean(temp_pts[:, 0])), float(np.mean(temp_pts[:, 1]))
                final_plane_model = plane_model
                break

        if table_cloud is None: return

        header = Header(stamp=ros_cloud_msg.header.stamp, frame_id=self.base_frame)

        # Output Data
        self.height_pub.publish(Float32(data=table_height))

        quat = self.get_surface_orientation(final_plane_model)
        pose_msg = PoseStamped(header=header)
        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = center_x, center_y, table_height
        pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z, pose_msg.pose.orientation.w = quat
        self.pose_pub.publish(pose_msg)

        # TF Broadcast pose of the table
        t = TransformStamped()
        t.header, t.child_frame_id = header, 'table_frame'
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = center_x, center_y, table_height
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)

        # Visuals
        table_cloud.paint_uniform_color([0, 1, 0]) 
        self.publish_o3d(table_cloud, self.table_cloud_pub, header)
        self.publish_height_markers(center_x, center_y, table_height, header)

    # helpers

    def get_surface_orientation(self, plane_model):
        normal = np.array(plane_model[:3])
        normal /= np.linalg.norm(normal)
        if normal[2] < 0: normal = -normal
        ref_x = np.array([1, 0, 0])
        target_y = np.cross(normal, ref_x)
        target_y /= np.linalg.norm(target_y)
        target_x = np.cross(target_y, normal)
        R = np.array([target_x, target_y, normal]).T
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2; qw = 0.25 * S; qx = (R[2,1] - R[1,2]) / S; qy = (R[0,2] - R[2,0]) / S; qz = (R[1,0] - R[0,1]) / S 
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2; qw = (R[2,1] - R[1,2]) / S; qx = 0.25 * S; qy = (R[0,1] + R[1,0]) / S; qz = (R[0,2] + R[2,0]) / S
        elif (R[1,1] > R[2,2]):
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2; qw = (R[0,2] - R[2,0]) / S; qx = 0.25 * S; qy = (R[0,1] + R[1,0]) / S; qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2; qw = (R[1,0] - R[0,1]) / S; qx = (R[0,2] + R[2,0]) / S; qy = (R[1,2] + R[2,1]) / S; qz = 0.25 * S
        return [qx, qy, qz, qw]

    def publish_height_markers(self, x, y, z, header):
        ma = MarkerArray()
        p = Marker(header=header, ns="h", id=0, type=Marker.CYLINDER, action=Marker.ADD)
        p.pose.position.x, p.pose.position.y, p.pose.position.z = x, y, z/2.0
        p.scale.x = p.scale.y = 0.02; p.scale.z = z; p.color.r, p.color.a = 1.0, 1.0
        ma.markers.append(p)
        t = Marker(header=header, ns="h", id=1, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y+0.05, z/2.0
        t.text = f"Height:{z:.3f}m"; t.scale.z = 0.08; t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
        ma.markers.append(t)
        self.viz_pub.publish(ma)

    def transform_to_matrix(self, ts):
        t, q = ts.transform.translation, ts.transform.rotation
        x,y,z,w = q.x,q.y,q.z,q.w
        R = np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],[2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],[2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
        m = np.eye(4); m[:3,:3] = R; m[0,3], m[1,3], m[2,3] = t.x, t.y, t.z
        return m

    def convert_ros_to_o3d(self, cloud):
        try:
            p = np.array([[x,y,z] for x,y,z in pc2.read_points(cloud, field_names=("x","y","z"), skip_nans=True)])
            o = o3d.geometry.PointCloud(); o.points = o3d.utility.Vector3dVector(p)
            return o
        except: return None

    def publish_o3d(self, cloud, pub, header):
        p = np.asarray(cloud.points); c = np.ones((len(p), 3))
        pts = []
        for i in range(len(p)):
            rgb = (255 << 16) | (255 << 8) | 255
            if cloud.has_colors():
                clr = (np.asarray(cloud.colors)[i]*255).astype(int)
                rgb = (clr[0] << 16) | (clr[1] << 8) | clr[2]
            rgb_f = struct.unpack('f', struct.pack('I', rgb))[0]
            pts.append([p[i][0], p[i][1], p[i][2], rgb_f])
        f = [PointField(name='x', offset=0, datatype=7, count=1), PointField(name='y', offset=4, datatype=7, count=1), PointField(name='z', offset=8, datatype=7, count=1), PointField(name='rgb', offset=12, datatype=7, count=1)]
        pub.publish(pc2.create_cloud(header, f, pts))

def main(args=None):
    rclpy.init(args=args)
    node = TableHeightLifecycleNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()