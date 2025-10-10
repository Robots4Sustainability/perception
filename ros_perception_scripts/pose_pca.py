import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from transforms3d.quaternions import mat2quat

import message_filters
from cv_bridge import CvBridge
import numpy as np
import struct
import sensor_msgs_py.point_cloud2 as pc2

class PointCloudCropperNode(Node):
    def __init__(self):
        super().__init__('pointcloud_cropper_node')

        self.bridge = CvBridge()
        self.camera_intrinsics = None

        # Declare input_mode parameter
        self.declare_parameter('input_mode', 'realsense')
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value

        # Determine topics based on mode
        if input_mode == 'robot':
            depth_topic = '/camera/aligned_depth_to_color/image_raw'
            img_topic = '/camera/color/image_raw'
            info_topic = '/camera/color/camera_info'
        elif input_mode == 'realsense':
            depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
            img_topic = '/camera/camera/color/image_raw'
            info_topic = '/camera/camera/color/camera_info'
        else:
            self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'realsense'")
            depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
            img_topic = '/camera/camera/color/image_raw'
            info_topic = '/camera/camera/color/camera_info'


        self.get_logger().info(f"Using input mode: '{input_mode}' with topics: {img_topic}, {depth_topic}, {info_topic}")

        # Message filter subscribers
        color_sub = message_filters.Subscriber(self, Image, img_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        det_sub = message_filters.Subscriber(self, Detection2DArray, '/pickable_objects')
        info_sub = message_filters.Subscriber(self, CameraInfo, info_topic)

        ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, det_sub, info_sub],
            queue_size=10,
            slop=0.2
        )
        ts.registerCallback(self.sync_callback)

        # Publishers
        self.pc_pub = self.create_publisher(PointCloud2, '/cropped_pointcloud', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/object_pose', 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info('PointCloud Cropper Node with PCA and TF broadcasting started.')

    def sync_callback(self, color_msg, depth_msg, detection_msg, camera_info_msg):
        if not self.camera_intrinsics:
            self.camera_intrinsics = {
                'fx': camera_info_msg.k[0], 'fy': camera_info_msg.k[4],
                'cx': camera_info_msg.k[2], 'cy': camera_info_msg.k[5],
                'height': camera_info_msg.height, 'width': camera_info_msg.width
            }

        try:
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        all_colored_points = []

        for idx, detection in enumerate(detection_msg.detections):
            detected_class = detection.results[0].hypothesis.class_id

            cx = int(detection.bbox.center.position.x)
            cy = int(detection.bbox.center.position.y)
            w = int(detection.bbox.size_x)
            h = int(detection.bbox.size_y)

            xmin = max(cx - w // 2, 0)
            xmax = min(cx + w // 2, self.camera_intrinsics['width'])
            ymin = max(cy - h // 2, 0)
            ymax = min(cy + h // 2, self.camera_intrinsics['height'])

            if ymin >= ymax or xmin >= xmax:
                self.get_logger().warn(f"Skipping invalid bbox for '{detected_class}'")
                continue

            points_in_box_xyz = []
            
            for v in range(ymin, ymax):
                for u in range(xmin, xmax):
                    depth = depth_image[v, u]
                    if depth > 0:
                        z = depth / 1000.0
                        x = (u - self.camera_intrinsics['cx']) * z / self.camera_intrinsics['fx']
                        y = (v - self.camera_intrinsics['cy']) * z / self.camera_intrinsics['fy']
                        
                        points_in_box_xyz.append([x, y, z])
                        
                        color = color_image[v, u]
                        b, g, r = color
                        rgb = struct.unpack('f', struct.pack('I', (int(r) << 16) | (int(g) << 8) | int(b)))[0]
                        all_colored_points.append([x, y, z, rgb])

            if len(points_in_box_xyz) >= 3:
                points_np = np.array(points_in_box_xyz)
                centroid = np.mean(points_np, axis=0)
                centered_points = points_np - centroid
                
                _, _, vh = np.linalg.svd(centered_points, full_matrices=False)
                rotation_matrix = vh.T
                
                # Ensure a right-handed coordinate system
                if np.linalg.det(rotation_matrix) < 0:
                    rotation_matrix[:, -1] *= -1

                quat_wxyz = mat2quat(rotation_matrix)
                quat = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]] # to x,y,z,w

                # Publish PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = color_msg.header.frame_id
                pose_msg.pose.position.x = float(centroid[0])
                pose_msg.pose.position.y = float(centroid[1])
                pose_msg.pose.position.z = float(centroid[2])
                pose_msg.pose.orientation.x = float(quat[0])
                pose_msg.pose.orientation.y = float(quat[1])
                pose_msg.pose.orientation.z = float(quat[2])
                pose_msg.pose.orientation.w = float(quat[3])
                self.pose_pub.publish(pose_msg)

                # Broadcast TF
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = color_msg.header.frame_id
                t.child_frame_id = f'{detected_class}_frame_{idx}'
                t.transform.translation.x = float(centroid[0])
                t.transform.translation.y = float(centroid[1])
                t.transform.translation.z = float(centroid[2])
                t.transform.rotation.x = float(quat[0])
                t.transform.rotation.y = float(quat[1])
                t.transform.rotation.z = float(quat[2])
                t.transform.rotation.w = float(quat[3])
                self.tf_broadcaster.sendTransform(t)

                self.get_logger().info(
                    f"Published pose and TF for '{detected_class}' object {idx} with {len(points_in_box_xyz)} points."
                )

        if all_colored_points:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = color_msg.header.frame_id

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

