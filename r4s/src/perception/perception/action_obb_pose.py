import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from sensor_msgs.msg import PointCloud2, PointField, Image
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header

import message_filters
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2
from transforms3d.quaternions import mat2quat
from tf2_ros import TransformBroadcaster

from rclpy.qos import QoSProfile, ReliabilityPolicy

class PointOBBCloudCropperLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('point_obb_cloud_cropper_node')
        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('input_mode', 'robot')

        # Placeholders
        self.pc_pub = None
        self.pose_pub = None
        self.tf_broadcaster = None
        
        self.pc_sub = None
        self.img_sub = None
        self.det_sub = None
        self.ts = None

        self._is_active = False
        self.get_logger().info('PointCloud OBB Cropper Lifecycle Node Initialized (Unconfigured).')

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring PointCloud OBB Cropper Node...")
        
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
            self.pose_pub = self.create_lifecycle_publisher(PoseStamped, '/object_pose_screwdriver', 10)
            self.tf_broadcaster = TransformBroadcaster(self)

            # Match the rosbag's reliable QoS
            reliable_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

            # Message Filters Subscriptions
            self.pc_sub = message_filters.Subscriber(self, PointCloud2, pc_topic, qos_profile=reliable_qos)
            self.img_sub = message_filters.Subscriber(self, Image, img_topic, qos_profile=reliable_qos)
            self.det_sub = message_filters.Subscriber(self, Detection2DArray, '/detections', qos_profile=reliable_qos)

            # Synchronizer
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.pc_sub, self.det_sub, self.img_sub],
                queue_size=500, 
                slop=1.5  
            )
            self.ts.registerCallback(self.sync_callback)

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure OBB Cropper Node: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating PointCloud OBB Cropper Node...")
        super().on_activate(state)
        self._is_active = True
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating PointCloud OBB Cropper Node...")
        self._is_active = False
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up OBB Cropper resources...")
        
        self.destroy_publisher(self.pc_pub)
        self.destroy_publisher(self.pose_pub)
        
        # Destroy underlying ROS subscriptions in message_filters
        if self.pc_sub is not None:
            self.destroy_subscription(self.pc_sub.sub)
        if self.img_sub is not None:
            self.destroy_subscription(self.img_sub.sub)
        if self.det_sub is not None:
            self.destroy_subscription(self.det_sub.sub)

        self.pc_pub = self.pose_pub = self.tf_broadcaster = None
        self.pc_sub = self.img_sub = self.det_sub = self.ts = None

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down OBB Cropper Node...")
        if self.pc_pub is not None:
            self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    def sync_callback(self, cloud_msg, detection_msg, image_msg):
        if not self._is_active:
            return

        # If there are no detections, do nothing
        if not detection_msg.detections:
            return

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
            cx = detection.bbox.center.position.x
            cy = detection.bbox.center.position.y
            theta = detection.bbox.center.theta
            w = detection.bbox.size_x
            h = detection.bbox.size_y

            angle_deg = np.degrees(theta)
            rect = ((cx, cy), (w, h), angle_deg)
            box = cv2.boxPoints(rect)
            box = np.int32(box) 

            mask = np.zeros((pc_height, pc_width), dtype=np.uint8)
            cv2.fillPoly(mask, [box], 255)

            cropped_points = cloud_points[mask == 255]
            cropped_colors = color_image[mask == 255]

            # Drop invalid depth points
            valid_mask = ~np.isnan(cropped_points).any(axis=1)
            cropped_points = cropped_points[valid_mask]
            cropped_colors = cropped_colors[valid_mask]

            self.get_logger().info(f"Class '{detected_class}': Cropped {len(cropped_points)} valid 3D points.")

            if len(cropped_points) >= 3:
                for pt, color in zip(cropped_points, cropped_colors):
                    x, y, z = pt
                    b, g, r = color
                    rgb = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
                    all_colored_points.append([x, y, z, rgb])

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
    node = PointOBBCloudCropperLifecycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()