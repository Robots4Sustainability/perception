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


class PointCloudCropperNode(Node):
    def __init__(self):
        super().__init__('pointcloud_cropper_node')

        self.bridge = CvBridge()

        # Declare input_mode parameter only
        self.declare_parameter('input_mode', 'realsense')
        input_mode = self.get_parameter('input_mode').get_parameter_value().string_value

        # Determine topics based on mode
        if input_mode == 'robot':
            pc_topic = '/camera/depth/color/points'
            img_topic = '/camera/color/image_raw'
        elif input_mode == 'realsense':
            pc_topic = '/camera/camera/depth/color/points'
            img_topic = '/camera/camera/color/image_raw'
        else:
            self.get_logger().warn(f"Unknown input_mode '{input_mode}', defaulting to 'realsense'")
            pc_topic = '/camera/camera/depth/color/points'
            img_topic = '/camera/camera/color/image_raw'

        self.get_logger().info(f"Using input mode: '{input_mode}' with topics: {pc_topic}, {img_topic}")

        # Message filter subscribers
        pc_sub = message_filters.Subscriber(self, PointCloud2, pc_topic)
        img_sub = message_filters.Subscriber(self, Image, img_topic)
        det_sub = message_filters.Subscriber(self, Detection2DArray, '/detections')

        ts = message_filters.ApproximateTimeSynchronizer(
            [pc_sub, det_sub, img_sub],
            queue_size=10,
            slop=0.1
        )
        ts.registerCallback(self.sync_callback)

        self.prev_poses = {} # Store previous pose for smoothing per object index
        self.alpha_pos = 0.2 # low-pass filter factor for position (0.2 = slow/smooth, 1.0 = no smoothing)
        self.alpha_rot = 0.1 # low-pass filter factor for rotation

        self.get_logger().info('PointCloud Cropper Node with robust PCA starting...')

    def sync_callback(self, cloud_msg, detection_msg, image_msg):
        try:
            color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        pc_width = cloud_msg.width
        pc_height = cloud_msg.height

        # Read all points (expensive but robust)
        cloud_points = np.array([
            [x, y, z]
            for x, y, z in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=False)
        ]).reshape((pc_height, pc_width, 3))

        all_colored_points = []
        current_frame_poses = {}

        for idx, detection in enumerate(detection_msg.detections):
            detected_class = detection.results[0].hypothesis.class_id
            
            # 1. Extract ROI
            cx, cy = int(detection.bbox.center.position.x), int(detection.bbox.center.position.y)
            w, h = int(detection.bbox.size_x), int(detection.bbox.size_y)
            xmin, xmax = max(cx - w // 2, 0), min(cx + w // 2, pc_width)
            ymin, ymax = max(cy - h // 2, 0), min(cy + h // 2, pc_height)

            cropped_points = cloud_points[ymin:ymax, xmin:xmax, :].reshape(-1, 3)
            cropped_colors = color_image[ymin:ymax, xmin:xmax, :].reshape(-1, 3)

            # Basic NaN filtering
            valid_mask = ~np.isnan(cropped_points).any(axis=1)
            cropped_points = cropped_points[valid_mask]
            cropped_colors = cropped_colors[valid_mask]

            # 2. Outlier Removal (Z-Score)
            if len(cropped_points) > 10:
                mean = np.mean(cropped_points, axis=0)
                std = np.std(cropped_points, axis=0)
                # Filter points > 2 std devs away
                z_score = np.abs((cropped_points - mean) / (std + 1e-6))
                inlier_mask = (z_score < 2.0).all(axis=1)
                cropped_points = cropped_points[inlier_mask]
                cropped_colors = cropped_colors[inlier_mask]

            # Re-collect for visualization
            for pt, color in zip(cropped_points, cropped_colors):
                x, y, z = pt
                rgb = struct.unpack('f', struct.pack('I', (int(color[2]) << 16) | (int(color[1]) << 8) | int(color[0])))[0]
                all_colored_points.append([x, y, z, rgb])

            if len(cropped_points) < 10:
                continue

            # 3. PCA & Alignment
            centroid = np.mean(cropped_points, axis=0)
            centered = cropped_points - centroid
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            R = vh.T

            # 3a. Normal Alignment (Minor Axis Z should point to Camera)
            # Camera frame: Z is forward (positive).
            # Objects are in front, so Z > 0. Normal pointing "at" camera means Z component should be negative.
            # However, if we view surface, normal is usually out of surface.
            # Let's enforce Z component of Z-axis (R[2,2]) is negative (pointing back to origin).
            if R[2, 2] > 0:
                R[:, 2] *= -1
                R[:, 1] *= -1 # Maintain right-hand rule by flipping Y too

            # 3b. Direction Disambiguation (Major Axis X towards "heavy" side)
            # Project points onto X axis
            projected_x = centered @ R[:, 0]
            # Simple heuristic: Skewness. If skew > 0, tail is right, bulk is left.
            # We want X to point to bulk. So if skew > 0, flip X.
            skew = np.sum(projected_x ** 3)
            if skew > 0:
                R[:, 0] *= -1
                R[:, 1] *= -1 # Maintain RHR

            # 4. Temporal Smoothing
            prev = self.prev_poses.get(idx)
            
            if prev is not None:
                # Position Smoothing
                centroid = self.alpha_pos * centroid + (1 - self.alpha_pos) * prev['pos']
                
                # Rotation Smoothing (Quaternion Lerp)
                q_curr = mat2quat(R)
                q_prev = prev['quat']
                
                # Ensure shortest path (dot product > 0)
                dot = np.dot(q_curr, q_prev)
                if dot < 0:
                    q_curr = -q_curr
                
                q_smooth = self.alpha_rot * q_curr + (1 - self.alpha_rot) * q_prev
                q_smooth /= np.linalg.norm(q_smooth) # Normalize
                quat = q_smooth
                
                # Reconstruct R from smoothed quat for other uses if needed (omitted for speed)
            else:
                quat = mat2quat(R)

            # Store for next frame
            current_frame_poses[idx] = {'pos': centroid, 'quat': quat}

            # Publish
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = cloud_msg.header.frame_id
            pose_msg.pose.position.x = float(centroid[0])
            pose_msg.pose.position.y = float(centroid[1])
            pose_msg.pose.position.z = float(centroid[2])
            pose_msg.pose.orientation.x = float(quat[1]) # Transforms3d gives w,x,y,z. ROS needs x,y,z,w
            pose_msg.pose.orientation.y = float(quat[2])
            pose_msg.pose.orientation.z = float(quat[3])
            pose_msg.pose.orientation.w = float(quat[0])

            self.pose_pub.publish(pose_msg)
            
            # TF
            t = TransformStamped()
            t.header = pose_msg.header
            t.child_frame_id = f'object_frame_{idx}'
            t.transform.translation.x = pose_msg.pose.position.x
            t.transform.translation.y = pose_msg.pose.position.y
            t.transform.translation.z = pose_msg.pose.position.z
            t.transform.rotation = pose_msg.pose.orientation
            self.tf_broadcaster.sendTransform(t)

        self.prev_poses = current_frame_poses # Update state with only currently seen objects (avoids stale ghosts)

        if all_colored_points:
            # ... (Publish cropped cloud code matches existing structure)
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
