import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from std_msgs.msg import Float32


import message_filters
from cv_bridge import CvBridge
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2
from transforms3d.quaternions import mat2quat
from tf2_ros import TransformBroadcaster

# IMPORT QOS PROFILES
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

class PointCloudCropperNode(Node):
    def __init__(self):
        super().__init__('pointcloud_cropper_node')
        self.bridge = CvBridge()

        # ... (Parameter declarations remain the same) ...
        self.declare_parameter('input_mode', 'robot')
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value

        if input_mode == 'robot':
            pc_topic = '/camera/depth/color/points'
            img_topic = '/camera/color/image_raw'
        else:
            pc_topic = '/camera/camera/depth/color/points'
            img_topic = '/camera/camera/color/image_raw'

        self.get_logger().info(f"Using topics: {pc_topic}, {img_topic}")

        # --- THE QOS FIX ---
        # Camera data usually requires Sensor Data QoS (Best Effort)
        self.pc_sub = message_filters.Subscriber(
            self, PointCloud2, pc_topic, qos_profile=qos_profile_sensor_data)
        
        self.img_sub = message_filters.Subscriber(
            self, Image, img_topic, qos_profile=qos_profile_sensor_data)
        
        # Detections are published by your YOLO node with default (Reliable) QoS
        # We explicitly define a reliable profile here to match it.
        reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.det_sub = message_filters.Subscriber(
            self, Detection2DArray, '/detections', qos_profile=reliable_qos)

        # --- THE SYNCHRONIZER FIX ---
        # Queue size of 100 ensures the node holds onto 3 seconds of point clouds (at 30fps)
        # while waiting for the YOLO node to finish processing the image.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.pc_sub, self.det_sub, self.img_sub],
            queue_size=100, 
            slop=0.05       # 50ms slop is plenty since YOLO passes the original timestamp
        )
        self.ts.registerCallback(self.sync_callback)

        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, '/cropped_pointcloud', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/object_sphere_marker', 10) # NEW PUBLISHER
        self.radius_pub = self.create_publisher(Float32, '/perception/detected_object_radius', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('PointCloud Cropper Node started (Waiting for Synced Data...).')

    def sync_callback(self, cloud_msg, detection_msg, image_msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        pc_width = cloud_msg.width
        pc_height = cloud_msg.height

        cloud_points = np.array([
            [x, y, z]
            for x, y, z in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False)
        ]).reshape((pc_height, pc_width, 3))

        all_colored_points = []

        for idx, detection in enumerate(detection_msg.detections):
            w_2d = detection.bbox.size_x
            h_2d = detection.bbox.size_y
            max_dist_2d = max(w_2d, h_2d)

            detected_class = detection.results[0].hypothesis.class_id

            # No class filtering here anymore

            cx = int(detection.bbox.center.position.x)
            cy = int(detection.bbox.center.position.y)
            w = int(detection.bbox.size_x)
            h = int(detection.bbox.size_y)

            xmin = max(cx - w // 2, 0)
            xmax = min(cx + w // 2, pc_width)
            ymin = max(cy - h // 2, 0)
            ymax = min(cy + h // 2, pc_height)

            cropped_points = cloud_points[ymin:ymax, xmin:xmax, :].reshape(-1, 3)
            cropped_colors = color_image[ymin:ymax, xmin:xmax, :].reshape(-1, 3)

            valid_mask = ~np.isnan(cropped_points).any(axis=1)
            cropped_points = cropped_points[valid_mask]
            cropped_colors = cropped_colors[valid_mask]

            for pt, color in zip(cropped_points, cropped_colors):
                x, y, z = pt
                b, g, r = color
                rgb = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
                all_colored_points.append([x, y, z, rgb])

            self.get_logger().info(
                f"Cropped '{detected_class}' object {idx}: [{xmin}:{xmax}, {ymin}:{ymax}] -> {len(cropped_points)} valid points"
            )

            if len(cropped_points) >= 3:
                centroid = np.mean(cropped_points, axis=0)
                centered = cropped_points - centroid
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                R = vh.T

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = centroid

                quat_wxyz = mat2quat(T[:3, :3])
                quat = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = cloud_msg.header.frame_id
                pose_msg.pose.position.x = float(centroid[0])
                pose_msg.pose.position.y = float(centroid[1])
                pose_msg.pose.position.z = float(centroid[2])
                pose_msg.pose.orientation.x = float(quat[0])
                pose_msg.pose.orientation.y = float(quat[1])
                pose_msg.pose.orientation.z = float(quat[2])
                pose_msg.pose.orientation.w = float(quat[3])

                self.pose_pub.publish(pose_msg)

                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = cloud_msg.header.frame_id
                t.child_frame_id = f'object_frame_{idx}'
                t.transform.translation.x = float(centroid[0])
                t.transform.translation.y = float(centroid[1])
                t.transform.translation.z = float(centroid[2])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])

                self.tf_broadcaster.sendTransform(t)
                self.get_logger().info(f"Published pose and TF for '{detected_class}' object {idx}")

                # --- SPHERE MARKER LOGIC ---
                # 1. Calculate the REAL 3D diameter using the cropped point cloud
                # This guarantees the sphere is measured in actual meters, not pixels!
                distances = np.linalg.norm(cropped_points - centroid, axis=1)
                radius = np.max(distances)

                radius_msg = Float32()
                metric_radius = float(np.percentile(distances, 98))
                radius_msg.data = metric_radius
                self.radius_pub.publish(radius_msg)

                sphere_diameter = float(metric_radius * 2.0)

                self.get_logger().info(f"Published METRIC radius: {metric_radius:.4f}m")

                # 2. Create the Marker
                marker = Marker()
                marker.header = cloud_msg.header
                marker.ns = "object_spheres"
                marker.id = idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                # 3. Set Position
                marker.pose.position.x = float(centroid[0])
                marker.pose.position.y = float(centroid[1])
                marker.pose.position.z = float(centroid[2])
                
                # CRITICAL FIX: RViz needs a valid quaternion, otherwise it might hide the marker
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                
                # 4. Set Scale to our calculated 3D diameter
                marker.scale.x = sphere_diameter
                marker.scale.y = sphere_diameter
                marker.scale.z = sphere_diameter
                
                # 5. Set Color (Green, semi-transparent)
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.5 
                
                self.marker_pub.publish(marker)

        if all_colored_points:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = cloud_msg.header.frame_id

            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            cropped_pc = pc2.create_cloud(header, fields, all_colored_points)
            self.pc_pub.publish(cropped_pc)


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudCropperNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()


#### FOR CAR_OBJECT_DETECTION_NODE
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2, PointField, Image
# from vision_msgs.msg import Detection2DArray
# from geometry_msgs.msg import PoseStamped, TransformStamped
# from std_msgs.msg import Header

# import message_filters
# from cv_bridge import CvBridge
# import numpy as np
# import struct
# import sensor_msgs_py.point_cloud2 as pc2
# from transforms3d.quaternions import mat2quat
# from tf2_ros import TransformBroadcaster

# # IMPORT QOS PROFILES
# from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

# class PointCloudCropperNode(Node):
#     def __init__(self):
#         super().__init__('pointcloud_cropper_node')
#         self.bridge = CvBridge()

#         # Storage for the single-shot detection
#         self.latest_detections = None

#         # Fixed parameter default to match your YOLO node
#         self.declare_parameter('input_mode', 'robot')
#         input_mode = self.get_parameter('input_mode').get_parameter_value().string_value

#         if input_mode == 'robot':
#             pc_topic = '/camera/depth/color/points'
#             img_topic = '/camera/color/image_raw'
#         else:
#             pc_topic = '/camera/camera/depth/color/points'
#             img_topic = '/camera/camera/color/image_raw'

#         self.get_logger().info(f"Using topics: {pc_topic}, {img_topic}")

#         # 1. Separate Subscriber for Detections (RELIABLE QoS)
#         reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
#         self.det_sub = self.create_subscription(
#             Detection2DArray, '/detections', self.detection_callback, reliable_qos)

#         # 2. Time Synchronizer ONLY for the live camera feeds (SENSOR DATA QoS)
#         self.pc_sub = message_filters.Subscriber(
#             self, PointCloud2, pc_topic, qos_profile=qos_profile_sensor_data)
        
#         self.img_sub = message_filters.Subscriber(
#             self, Image, img_topic, qos_profile=qos_profile_sensor_data)
        
#         # Loosened slop to 0.2 to allow for depth/rgb lag
#         self.ts = message_filters.ApproximateTimeSynchronizer(
#             [self.pc_sub, self.img_sub], queue_size=30, slop=0.2 
#         )
#         self.ts.registerCallback(self.sync_callback)

#         # Publishers
#         self.pc_pub = self.create_publisher(PointCloud2, '/cropped_pointcloud', 10)
#         self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
#         self.tf_broadcaster = TransformBroadcaster(self)

#         self.get_logger().info('PointCloud Cropper Node started (Waiting for single YOLO detection...).')

#     def detection_callback(self, msg):
#         # Save the detections to memory
#         self.latest_detections = msg.detections
#         self.get_logger().info(f"Received {len(self.latest_detections)} detections! Cropper is now active.")

#     # NOTICE: detection_msg has been completely removed from this signature!
#     def sync_callback(self, cloud_msg, image_msg):
#         # Guard check: Don't do anything until detections are received
#         if not self.latest_detections:
#             return

#         try:
#             color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
#         except Exception as e:
#             self.get_logger().error(f"Image conversion error: {e}")
#             return

#         pc_width = cloud_msg.width
#         pc_height = cloud_msg.height

#         cloud_points = np.array([
#             [x, y, z]
#             for x, y, z in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False)
#         ]).reshape((pc_height, pc_width, 3))

#         all_colored_points = []

#         # Iterate over the SAVED detections
#         for idx, detection in enumerate(self.latest_detections):
#             detected_class = detection.results[0].hypothesis.class_id

#             cx = int(detection.bbox.center.position.x)
#             cy = int(detection.bbox.center.position.y)
#             w = int(detection.bbox.size_x)
#             h = int(detection.bbox.size_y)

#             xmin = max(cx - w // 2, 0)
#             xmax = min(cx + w // 2, pc_width)
#             ymin = max(cy - h // 2, 0)
#             ymax = min(cy + h // 2, pc_height)

#             cropped_points = cloud_points[ymin:ymax, xmin:xmax, :].reshape(-1, 3)
#             cropped_colors = color_image[ymin:ymax, xmin:xmax, :].reshape(-1, 3)

#             valid_mask = ~np.isnan(cropped_points).any(axis=1)
#             cropped_points = cropped_points[valid_mask]
#             cropped_colors = cropped_colors[valid_mask]

#             for pt, color in zip(cropped_points, cropped_colors):
#                 x, y, z = pt
#                 b, g, r = color
#                 rgb = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
#                 all_colored_points.append([x, y, z, rgb])

#             # Print out the results to terminal so you know it's working
#             self.get_logger().info(
#                 f"Cropped '{detected_class}' object {idx}: [{xmin}:{xmax}, {ymin}:{ymax}] -> {len(cropped_points)} valid points"
#             )

#             if len(cropped_points) >= 3:
#                 centroid = np.mean(cropped_points, axis=0)
#                 centered = cropped_points - centroid
#                 _, _, vh = np.linalg.svd(centered, full_matrices=False)
#                 R = vh.T

#                 T = np.eye(4)
#                 T[:3, :3] = R
#                 T[:3, 3] = centroid

#                 quat_wxyz = mat2quat(T[:3, :3])
#                 quat = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

#                 pose_msg = PoseStamped()
#                 pose_msg.header.stamp = self.get_clock().now().to_msg()
#                 pose_msg.header.frame_id = cloud_msg.header.frame_id
#                 pose_msg.pose.position.x = float(centroid[0])
#                 pose_msg.pose.position.y = float(centroid[1])
#                 pose_msg.pose.position.z = float(centroid[2])
#                 pose_msg.pose.orientation.x = float(quat[0])
#                 pose_msg.pose.orientation.y = float(quat[1])
#                 pose_msg.pose.orientation.z = float(quat[2])
#                 pose_msg.pose.orientation.w = float(quat[3])

#                 self.pose_pub.publish(pose_msg)

#                 t = TransformStamped()
#                 t.header.stamp = self.get_clock().now().to_msg()
#                 t.header.frame_id = cloud_msg.header.frame_id
#                 t.child_frame_id = f'object_frame_{idx}'
#                 t.transform.translation.x = float(centroid[0])
#                 t.transform.translation.y = float(centroid[1])
#                 t.transform.translation.z = float(centroid[2])
#                 t.transform.rotation.x = float(quat[0])
#                 t.transform.rotation.y = float(quat[1])
#                 t.transform.rotation.z = float(quat[2])
#                 t.transform.rotation.w = float(quat[3])

#                 self.tf_broadcaster.sendTransform(t)

#         if all_colored_points:
#             header = Header()
#             header.stamp = self.get_clock().now().to_msg()
#             header.frame_id = cloud_msg.header.frame_id

#             fields = [
#                 PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#                 PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#                 PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
#                 PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
#             ]

#             cropped_pc = pc2.create_cloud(header, fields, all_colored_points)
#             self.pc_pub.publish(cropped_pc)

# def main(args=None):
#     rclpy.init(args=args)
#     node = PointCloudCropperNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()