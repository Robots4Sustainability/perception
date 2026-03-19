import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2, PointField, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header, Float32
from visualization_msgs.msg import Marker

import message_filters
from cv_bridge import CvBridge
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2
from transforms3d.quaternions import mat2quat
from tf2_ros import TransformBroadcaster

from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy

class PointCloudCropperLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('pointcloud_cropper_node')
        self.bridge = CvBridge()

        self.declare_parameter('input_mode', 'robot')
        self.declare_parameter('target_class', '')

        self.pc_pub = None
        self.pose_pub = None
        self.marker_pub = None
        self.radius_pub = None
        self.tf_broadcaster = None
        
        self.pc_sub = None
        self.img_sub = None
        self.det_sub = None
        self.ts = None

        self._is_active = False
        self.get_logger().info('PointCloud Cropper Lifecycle Node Initialized (Unconfigured).')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring PointCloud Cropper Node...")
        
        try:
            input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
            if input_mode == 'robot':
                pc_topic = '/camera/depth/color/points'
                img_topic = '/camera/color/image_raw'
            else:
                pc_topic = '/camera/camera/depth/color/points'
                img_topic = '/camera/camera/color/image_raw'

            # Lifecycle Publishers
            self.pc_pub = self.create_lifecycle_publisher(PointCloud2, '/cropped_pointcloud', 10)
            self.pose_pub = self.create_lifecycle_publisher(PoseStamped, '/object_pose', 10)
            self.marker_pub = self.create_lifecycle_publisher(Marker, '/object_sphere_marker', 10)
            self.radius_pub = self.create_lifecycle_publisher(Float32, '/perception/detected_object_radius', 10)
            self.tf_broadcaster = TransformBroadcaster(self)

            # Subscriptions
            self.pc_sub = message_filters.Subscriber(self, PointCloud2, pc_topic, qos_profile=qos_profile_sensor_data)
            self.img_sub = message_filters.Subscriber(self, Image, img_topic, qos_profile=qos_profile_sensor_data)
            reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            self.det_sub = message_filters.Subscriber(self, Detection2DArray, '/detections', qos_profile=reliable_qos)

            # Synchronizer
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.pc_sub, self.det_sub, self.img_sub], queue_size=100, slop=0.05
            )
            self.ts.registerCallback(self.sync_callback)

            self.target_class = self.get_parameter('target_class').get_parameter_value().string_value

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure Cropper Node: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating PointCloud Cropper Node...")
        super().on_activate(state)
        self._is_active = True
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating PointCloud Cropper Node...")
        self._is_active = False
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up Cropper resources...")
        
        self.destroy_publisher(self.pc_pub)
        self.destroy_publisher(self.pose_pub)
        self.destroy_publisher(self.marker_pub)
        self.destroy_publisher(self.radius_pub)
        
        if self.pc_sub is not None: self.destroy_subscription(self.pc_sub.sub)
        if self.img_sub is not None: self.destroy_subscription(self.img_sub.sub)
        if self.det_sub is not None: self.destroy_subscription(self.det_sub.sub)

        self.pc_pub = self.pose_pub = self.marker_pub = self.radius_pub = self.tf_broadcaster = None
        self.pc_sub = self.img_sub = self.det_sub = self.ts = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        if self.pc_pub is not None:
            self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def sync_callback(self, cloud_msg, detection_msg, image_msg):
        if not self._is_active: return

        target = self.get_parameter('target_class').get_parameter_value().string_value

        try:
            color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        pc_width, pc_height = cloud_msg.width, cloud_msg.height
        cloud_points = np.array([
            [x, y, z] for x, y, z in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False)
        ]).reshape((pc_height, pc_width, 3))

        all_colored_points = []

        for idx, detection in enumerate(detection_msg.detections):
            detected_class = detection.results[0].hypothesis.class_id

            if target != '' and detected_class != target:
                self.get_logger().info(f"Ignoring '{detected_class}', searching for '{target}'...")
                continue

            cx, cy = int(detection.bbox.center.position.x), int(detection.bbox.center.position.y)
            w, h = int(detection.bbox.size_x), int(detection.bbox.size_y)

            xmin, xmax = max(cx - w // 2, 0), min(cx + w // 2, pc_width)
            ymin, ymax = max(cy - h // 2, 0), min(cy + h // 2, pc_height)

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
                
                if detected_class in ["motor", "motor_grip"]:
                    # Fixed rotation: 90 degrees clockwise around Z-axis
                    # Old Frame: X=Right (Red), Y=Down (Green), Z=Forward (Blue)
                    # New Frame: X=Down (old +Y), Y=Left (old -X), Z=Forward
                    R = np.array([
                        [ 0.0, -1.0,  0.0],
                        [ 1.0,  0.0,  0.0],
                        [ 0.0,  0.0,  1.0]
                    ])
                    self.get_logger().info(f"Using FIXED 90-deg CW rotation for {detected_class}")
                
                elif detected_class == "unit":
                    # Fixed rotation: 30 degrees clockwise around Z-axis
                    theta = np.radians(45)
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([
                        [ c, -s, 0.0],
                        [ s,  c, 0.0],
                        [0.0, 0.0, 1.0]
                    ])
                    self.get_logger().info(f"Using FIXED 30-deg rotation for {detected_class}")

                else:
                    # Standard PCA rotation for all other objects
                    centered = cropped_points - centroid
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    R = vh.T
                    # Ensure a right-handed coordinate system
                    if np.linalg.det(R) < 0:
                        R[:, 2] *= -1
                    self.get_logger().info(f"Using PCA rotation for {detected_class}")

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = centroid
                quat_wxyz = mat2quat(T[:3, :3])
                quat = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]

                # Pose Publish
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

                # TF Publish
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

                # Radius & Sphere Logic
                distances = np.linalg.norm(cropped_points - centroid, axis=1)
                metric_radius = float(np.percentile(distances, 98))

                radius_msg = Float32()
                radius_msg.data = metric_radius
                self.radius_pub.publish(radius_msg)

                sphere_diameter = metric_radius * 2.0
                marker = Marker()
                marker.header = cloud_msg.header
                marker.ns = "object_spheres"
                marker.id = idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(centroid[0])
                marker.pose.position.y = float(centroid[1])
                marker.pose.position.z = float(centroid[2])
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0
                marker.scale.x = sphere_diameter
                marker.scale.y = sphere_diameter
                marker.scale.z = sphere_diameter
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
    node = PointCloudCropperLifecycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()