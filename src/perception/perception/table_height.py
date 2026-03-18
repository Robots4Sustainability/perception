import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32, Header
from geometry_msgs.msg import PointStamped

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs 

import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import struct


class TableHeightNode(Node):
    def __init__(self):
        super().__init__('table_height_node')

        self.declare_parameter('input_mode', 'robot')
        # The frame that represents the floor (eddie_base_footprint?)
        self.declare_parameter('base_frame', 'eddie_base_footprint') 
        
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value

        if input_mode == 'robot':
            pc_topic = '/camera/depth/color/points'
        else:
            pc_topic = '/camera/camera/depth/color/points'

        self.get_logger().info(f"Subscribing to: {pc_topic}")
        self.get_logger().info(f"Calculating height relative to frame: {self.base_frame}")

        self.pc_sub = self.create_subscription(PointCloud2, pc_topic, self.cloud_callback, 10)
        
        # Publisher to output the height
        self.height_pub = self.create_publisher(Float32, '/perception/table_height', 10)

        self.table_cloud_pub = self.create_publisher(PointCloud2, '/perception/debug/table_height_plane', 10)

        # setup TF2 Listener
        # looks up where the camera is relative to the robot base
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def cloud_callback(self, ros_cloud_msg):
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
        debug_header = Header()
        debug_header.stamp = ros_cloud_msg.header.stamp
        debug_header.frame_id = self.base_frame

        table_cloud.paint_uniform_color([0, 1, 0]) 
        self.publish_o3d(table_cloud, self.table_cloud_pub, debug_header)

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
    node = TableHeightNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()