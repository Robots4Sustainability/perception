import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Header
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped, PoseStamped
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct


class TableHeightRansac(LifecycleNode):
    def __init__(self):
        super().__init__('table_height_estimator')

        self.declare_parameter('input_mode', 'robot')
        # The frame that represents the floor (eddie_base_footprint?)
        self.declare_parameter('base_frame', 'eddie_base_footprint') 
        
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        if input_mode == 'robot':
            pc_topic = '/camera/depth/color/points'
        else:
            pc_topic = '/camera/camera/depth/color/points'

        # Lifecycle State Flag
        self._is_active = False

        # TF2 Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info("Table Height Lifecycle Node Initialized (Unconfigured).")

        '''
        
        self.get_logger().info(f"Subscribing to: {pc_topic}")
        self.get_logger().info(f"Calculating height relative to frame: {self.base_frame}")

        self.pc_sub = self.create_subscription(PointCloud2, pc_topic, self.cloud_callback, 10)
        
        # Publishers to output the height and pose of table
        self.height_pub = self.create_publisher(Float32, '/perception/table_height', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/perception/table_pose', 10)

        # publishers for rviz
        self.table_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/table_height_plane', 10)
        self.viz_pub = self.create_publisher(MarkerArray, '/perception/debug/table_height_viz', 10)

        # setup TF2 Listener
        # looks up where the camera is relative to the robot base
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        '''

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring Table Height Estimator...")
        
        # Initialize Publishers
        self.height_pub = self.create_lifecycle_publisher(Float32, '/table_height_value', 10)
        self.pose_pub = self.create_lifecycle_publisher(PoseStamped, '/perception/table_pose', 10)
        self.table_cloud_pub = self.create_lifecycle_publisher(PointCloud2, '/perception/debug/table_height_plane', 10)
        self.viz_pub = self.create_lifecycle_publisher(MarkerArray, '/perception/debug/table_height_viz', 10)

        # Initialize Subscriber
        self.pc_sub = self.create_subscription(PointCloud2, self.pc_topic, self.cloud_callback, 10)

        self.get_logger().info("Successfully Configured.")
        return TransitionCallbackReturn.SUCCESS

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
        self.destroy_publisher(self.height_pub)
        self.destroy_publisher(self.pose_pub)
        self.destroy_publisher(self.table_cloud_pub)
        self.destroy_publisher(self.viz_pub)
        return TransitionCallbackReturn.SUCCESS
    
    def cloud_callback(self, ros_cloud_msg):
        if not self._is_active:
            return
        
        # Look up the transform from camera to base
        try:
            transform_stamped = self.tf_buffer.lookup_transform(
                self.base_frame,             
                ros_cloud_msg.header.frame_id, 
                rclpy.time.Time()            
            )
        except TransformException as ex:
            self.get_logger().warn(f"TF Error: {ex}")
            return

        pcd = self.convert_ros_to_o3d(ros_cloud_msg)
        if pcd is None or len(pcd.points) == 0: return

        # transform the whole cloud to the base frame so that we can analyze it relative to the robot's floor
        # z is up relative to gravity
        tf_matrix = self.transform_to_matrix(transform_stamped)
        pcd.transform(tf_matrix)

        pcd_down = pcd.voxel_down_sample(voxel_size=0.01)

        # iterative ransac until we find a horizontal plane that is above the floor height threshold
        # (reject wall and floor planes)
        table_cloud = None
        table_height = 0.0
        center_x = 0.0
        center_y = 0.0
        final_plane_model = None
        
        for attempt in range(5): # Allow up to 5 planes to be checked
            try:
                plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.015,
                                                            ransac_n=3,
                                                            num_iterations=1000)
            except Exception:
                break 

            temp_cloud = pcd_down.select_by_index(inliers)
            temp_pts = np.asarray(temp_cloud.points)
            
            # Normal Vector [a, b, c]
            a, b, c, d = plane_model
            
            # c represents how much the normal points along the z axis.
            # abs(c) = 1.0 = perfectly flat
            # abs(c) = 0.0 = perfectly vertical
            # > 0.85 allows for up to ~30 degrees of tilt.
            is_horizontal = abs(c) > 0.85 
            
            # Since the cloud is in the base frame, the Z coordinate is the physical height
            avg_z = np.mean(temp_pts[:, 2])

            if not is_horizontal:
                # self.get_logger().info("Ignoring wall")
                pcd_down = pcd_down.select_by_index(inliers, invert=True)
                continue
                
            elif avg_z < 0.2: # Assuming table is higher than 20cm
                # self.get_logger().info(f"Ignoring the floor at height {avg_z:.2f}m...")
                pcd_down = pcd_down.select_by_index(inliers, invert=True)
                continue
                
            else:
                # horizontal plane that is not the floor (table hopefully)
                table_cloud = temp_cloud
                table_height = avg_z

                # Save the X,Y center for visualization line
                center_x = float(np.mean(temp_pts[:, 0]))
                center_y = float(np.mean(temp_pts[:, 1]))

                final_plane_model = plane_model

                break

        if table_cloud is None or len(table_cloud.points) == 0:
            return 

        # publish height
        msg = Float32()
        msg.data = float(table_height)
        self.height_pub.publish(msg)
        self.get_logger().info(f"Table Height from '{self.base_frame}': {table_height:.3f} meters")

        # Visualize the table in rviz
        # (Since we transformed the cloud, tell RViz this cloud is in the base_frame)
        header = Header()
        header.stamp = ros_cloud_msg.header.stamp
        header.frame_id = self.base_frame

        # publish pose
        quat = self.get_surface_orientation(final_plane_model)
        
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.pose.position.x = center_x
        pose_msg.pose.position.y = center_y
        pose_msg.pose.position.z = table_height
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        
        self.pose_pub.publish(pose_msg)

        t = TransformStamped()
        t.header = header
        t.child_frame_id = 'table_frame' # The frame name for the robot to move to
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = center_x, center_y, table_height
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)

        # Visualize Cloud and Markers
        table_cloud.paint_uniform_color([0, 1, 0]) 
        self.publish_o3d(table_cloud, self.table_cloud_pub, header)
        self.publish_height_markers(center_x, center_y, table_height, header)

    def get_surface_orientation(self, plane_model):
        """
        Creates a Quaternion where the Z-Axis points perfectly up out of the table surface.
        """
        normal = np.array(plane_model[:3])
        normal = normal / np.linalg.norm(normal)

        # Ensure the normal vector points up (+z).
        if normal[2] < 0:
            target_z = -normal
        else:
            target_z = normal

        # Construct orthogonal axes 
        # X and Y flat on the table, Z pointing up
        # use the world X-axis as a reference direction for the table's X-axis
        ref_x = np.array([1, 0, 0])
        
        # Y = Z cross X
        target_y = np.cross(target_z, ref_x)
        target_y = target_y / np.linalg.norm(target_y)

        # Recompute X = Y cross Z (ensure perfectly 90 degrees)
        target_x = np.cross(target_y, target_z)
        target_x = target_x / np.linalg.norm(target_x)

        # Rotation Matrix [ X  Y  Z ]
        R = np.array([target_x, target_y, target_z]).T

        # Convert to Quaternion [x, y, z, w]
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2; qw = 0.25 * S; qx = (R[2,1] - R[1,2]) / S; qy = (R[0,2] - R[2,0]) / S; qz = (R[1,0] - R[0,1]) / S 
        elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2; qw = (R[2,1] - R[1,2]) / S; qx = 0.25 * S; qy = (R[0,1] + R[1,0]) / S; qz = (R[0,2] + R[2,0]) / S
        elif (R[1,1] > R[2,2]):
            S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2; qw = (R[0,2] - R[2,0]) / S; qx = (R[0,1] + R[1,0]) / S; qy = 0.25 * S; qz = (R[1,2] + R[2,1]) / S
        else:
            S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2; qw = (R[1,0] - R[0,1]) / S; qx = (R[0,2] + R[2,0]) / S; qy = (R[1,2] + R[2,1]) / S; qz = 0.25 * S

        return[qx, qy, qz, qw]

    def publish_height_markers(self, x, y, z, header):
        marker_array = MarkerArray()

        # vertical cylinder
        pole = Marker()
        pole.header = header
        pole.ns = "table_height"
        pole.id = 0
        pole.type = Marker.CYLINDER
        pole.action = Marker.ADD
        
        # Center of the cylinder is halfway between floor and table
        pole.pose.position.x = x
        pole.pose.position.y = y
        pole.pose.position.z = z / 2.0 
        pole.pose.orientation.w = 1.0
        
        pole.scale.x = 0.02
        pole.scale.y = 0.02
        pole.scale.z = z
        
        pole.color.r = 1.0
        pole.color.g = 0.0
        pole.color.b = 0.0
        pole.color.a = 1.0
        
        marker_array.markers.append(pole)

        # test label
        text = Marker()
        text.header = header
        text.ns = "table_height"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        
        text.pose.position.x = x
        text.pose.position.y = y + 0.05
        text.pose.position.z = z / 2.0
        text.pose.orientation.w = 1.0
        
        text.text = f"Height:{z:.3f}m"
        text.scale.x = 0.05
        text.scale.z = 0.08
        
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        
        marker_array.markers.append(text)

        self.viz_pub.publish(marker_array)

    def transform_to_matrix(self, transform_stamped):
        """Converts a ROS TransformStamped into a 4x4 numpy transformation matrix."""
        t = transform_stamped.transform.translation
        q = transform_stamped.transform.rotation

        # Quaternion to Rotation Matrix
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([
                    [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                    [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                    [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
                    ])

        # Build 4x4 Matrix
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[0, 3] = t.x
        matrix[1, 3] = t.y
        matrix[2, 3] = t.z
        return matrix


    def convert_ros_to_o3d(self, ros_cloud):
        try:
            pcd_np = np.array([[x, y, z] for x, y, z in pc2.read_points(ros_cloud, field_names=("x","y","z"), skip_nans=True)])
            if pcd_np.shape[0] == 0: return None
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
            return pcd
        except: return None

    def publish_o3d(self, o3d_cloud, publisher, header):
        points = np.asarray(o3d_cloud.points)
        if len(points) == 0: return
        if not o3d_cloud.has_colors(): colors = np.ones((len(points), 3))
        else: colors = np.asarray(o3d_cloud.colors)
        pts_w_color =[]
        for i in range(len(points)):
            x, y, z = points[i]; r, g, b = colors[i]
            rgb = (int(r*255) << 16) | (int(g*255) << 8) | int(b*255)
            rgb_f = struct.unpack('f', struct.pack('I', rgb))[0]
            pts_w_color.append([x, y, z, rgb_f])
        fields =[PointField(name='x', offset=0, datatype=7, count=1), PointField(name='y', offset=4, datatype=7, count=1), PointField(name='z', offset=8, datatype=7, count=1), PointField(name='rgb', offset=12, datatype=7, count=1)]
        publisher.publish(pc2.create_cloud(header, fields, pts_w_color))

def main(args=None):
    rclpy.init(args=args)
    node = TableHeightRansac()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()