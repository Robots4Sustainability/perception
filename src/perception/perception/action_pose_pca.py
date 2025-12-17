import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header

import message_filters
from cv_bridge import CvBridge
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2
from transforms3d.quaternions import mat2quat
from tf2_ros import TransformBroadcaster
from std_srvs.srv import SetBool  # <-- NEW: Service Import


class PointCloudCropperNode(Node):
    def __init__(self):
        super().__init__('pointcloud_cropper_node')

        self.bridge = CvBridge()

        # --- PARAMETERS ---
        self.declare_parameter('input_mode', 'realsense')
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value

        # Determine topics based on mode
        if input_mode == 'robot':
            self.pc_topic = '/camera/depth/color/points'
            self.img_topic = '/camera/color/image_raw'
        elif input_mode == 'realsense':
            self.pc_topic = '/camera/camera/depth/color/points'
            self.img_topic = '/camera/camera/color/image_raw'
        else:
            self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'realsense'")
            self.pc_topic = '/camera/camera/depth/color/points'
            self.img_topic = '/camera/camera/color/image_raw'

        self.get_logger().info(f"Using input mode: '{input_mode}' with topics: {self.pc_topic}, {self.img_topic}")

        # --- 1. SETUP PUBLISHERS (Always Active) ---
        self.pc_pub = self.create_publisher(PointCloud2, '/cropped_pointcloud', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- 2. OPTIMIZATION: INITIALIZE STATE ---
        # We do NOT create subscribers or synchronizers here.
        self.pc_sub = None
        self.img_sub = None
        self.det_sub = None
        self.ts = None
        self.is_active = False

        # --- 3. CREATE TOGGLE SERVICE ---
        self.srv = self.create_service(SetBool, 'toggle_pose', self.toggle_callback)

        self.get_logger().info('PointCloud Cropper Node Ready in STANDBY mode.')
        self.get_logger().info('Send "True" to /toggle_pose to start synchronization.')

    def toggle_callback(self, request, response):
        """Service to Turn Processing ON or OFF"""
        if request.data:  # ENABLE
            if not self.is_active:
                self.get_logger().info("ACTIVATING: Starting Time Synchronizer...")
                
                # Create the message filters
                self.pc_sub = message_filters.Subscriber(self, PointCloud2, self.pc_topic)
                self.img_sub = message_filters.Subscriber(self, Image, self.img_topic)
                self.det_sub = message_filters.Subscriber(self, Detection2DArray, '/detections')

                # Create the Synchronizer
                self.ts = message_filters.ApproximateTimeSynchronizer(
                    [self.pc_sub, self.det_sub, self.img_sub],
                    queue_size=10,
                    slop=0.1
                )
                self.ts.registerCallback(self.sync_callback)
                
                self.is_active = True
                response.message = "Pose Node Activated"
                response.success = True
            else:
                response.message = "Already Active"
                response.success = True

        else:  # DISABLE
            if self.is_active:
                self.get_logger().info("DEACTIVATING: Destroying Synchronizer to save CPU...")
                
                # message_filters.Subscriber wrappers don't have a clean 'destroy' method
                # exposed easily, but destroying the underlying subscription works.
                # However, dropping the reference to the synchronizer stops the callbacks.
                
                # Best practice: explicitly destroy the underlying rclpy subscriptions
                if self.pc_sub: self.destroy_subscription(self.pc_sub.sub)
                if self.img_sub: self.destroy_subscription(self.img_sub.sub)
                if self.det_sub: self.destroy_subscription(self.det_sub.sub)

                self.pc_sub = None
                self.img_sub = None
                self.det_sub = None
                self.ts = None
                
                self.is_active = False
                response.message = "Pose Node Deactivated"
                response.success = True
            else:
                response.message = "Already Inactive"
                response.success = True
        
        return response

    def sync_callback(self, cloud_msg, detection_msg, image_msg):
        # This callback logic remains EXACTLY the same as your original code
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
            detected_class = detection.results[0].hypothesis.class_id

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()