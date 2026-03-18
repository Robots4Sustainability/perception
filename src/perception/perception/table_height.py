import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Float32
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
        # the node looks up where the camera is relative to the robot base
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def cloud_callback(self, ros_cloud_msg):
        pcd = self.convert_ros_to_o3d(ros_cloud_msg)
        if pcd is None: return

        # Depth Cutoff
        points = np.asarray(pcd.points)
        if len(points) == 0: return
        mask = points[:, 2] < 1.5 
        pcd = pcd.select_by_index(np.where(mask)[0])
        
        pcd_down = pcd.voxel_down_sample(voxel_size=0.01)

        # reject walls (Iterative RANSAC)
        table_cloud = None
        
        for attempt in range(3): # Try up to 3 times to find the real table
            try:
                plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.01,
                                                            ransac_n=3,
                                                            num_iterations=1000)
            except Exception:
                break # No more planes found

            # Extract this plane
            temp_cloud = pcd_down.select_by_index(inliers)
            temp_pts = np.asarray(temp_cloud.points)
            
            # Calculate how far away this plane is from the camera
            avg_z = np.mean(temp_pts[:, 2])

            # If the plane is further than 0.85 meters, it could be wall
            # tune treshold based on tests
            # TODO: maybe also check the plane normal to see if it's vertical? (walls) vs horizontal? (table)
            if avg_z > 1.3: 
                self.get_logger().info(f"Ignoring a wall/floor at {avg_z:.2f}m away...")
                # Invert the selection to REMOVE the wall points, then loop again
                pcd_down = pcd_down.select_by_index(inliers, invert=True)
                continue
            else:
                # plane that is close to the camera is the table
                table_cloud = temp_cloud
                break

        # Safety check if we filtered out everything
        if table_cloud is None or len(table_cloud.points) == 0:
            return 

        # Visualize the Table in RViz
        table_cloud.paint_uniform_color([0, 1, 0]) # Paint it Green
        self.publish_o3d(table_cloud, self.table_cloud_pub, ros_cloud_msg.header)

        # Find the center point
        table_pts = np.asarray(table_cloud.points)
        center_x = np.mean(table_pts[:, 0])
        center_y = np.mean(table_pts[:, 1])
        center_z = np.mean(table_pts[:, 2])

        camera_point = PointStamped()
        camera_point.header = ros_cloud_msg.header
        camera_point.point.x = float(center_x)
        camera_point.point.y = float(center_y)
        camera_point.point.z = float(center_z)

        # Transform to Base Frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,             
                camera_point.header.frame_id, 
                rclpy.time.Time()            
            )

            base_point = tf2_geometry_msgs.do_transform_point(camera_point, transform)

            # Extract height
            table_height = base_point.point.z

            msg = Float32()
            msg.data = table_height
            self.height_pub.publish(msg)

            self.get_logger().info(f"Table Height from '{self.base_frame}': {table_height:.3f} meters")

        except TransformException as ex:
            self.get_logger().warn(f"TF Error: {ex}")


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