import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import struct
import math
import os
import torch
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

def get_package_name_from_path(file_path):
    p = Path(file_path)
    try:
        idx = p.parts.index('site-packages')
        return p.parts[idx + 1]
    except (ValueError, IndexError):
        return "perception" # Fallback to your package name

class BodyPoseEstimatorLifecycleNode(LifecycleNode):
    def __init__(self):
        super().__init__('subdoor_pose_estimator')
        self.package_name = get_package_name_from_path(__file__)
        
        # --- Parameters ---
        self.declare_parameter('model_path', '') 
        self.declare_parameter('input_mode', 'robot') 
        self.declare_parameter('conf_threshold', 0.25)
        
        # Placeholders for resources
        self.model = None
        self.bridge = CvBridge()
        self.model_path = ""
        self.input_mode = ""
        self.conf = 0.25

        self.marker_pub = None
        self.debug_pub = None
        self.img_sub = None
        self.pc_sub = None
        self.ts = None

        self._is_active = False
        self.latest_poses = [] 
        
        self.get_logger().info("Body Pose Estimator Lifecycle Node Initialized (Unconfigured).")

    # --- LIFECYCLE CALLBACKS ---

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring Body Pose Estimator Node...")
        
        try:
            param_model_path = self.get_parameter('model_path').get_parameter_value().string_value
            self.input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
            self.conf = self.get_parameter('conf_threshold').get_parameter_value().double_value
            
            # Resolve Model Path
            if param_model_path:
                self.model_path = param_model_path
            else:
                share_dir = get_package_share_directory(self.package_name)
                self.model_path = os.path.join(share_dir, 'models', 'subdoor.pt')

            # Load YOLO Model
            self.get_logger().info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)

            # --- Topics ---
            img_topic = '/camera/color/image_raw' if self.input_mode == 'robot' else '/camera/camera/color/image_raw'
            pc_topic = '/camera/depth/color/points' if self.input_mode == 'robot' else '/camera/camera/depth/color/points'

            # --- Lifecycle Publishers ---
            self.marker_pub = self.create_lifecycle_publisher(MarkerArray, '/body_markers', 10)
            self.debug_pub = self.create_lifecycle_publisher(Image, '/body_debug_image', 10)

            # --- Synchronization Subscriptions ---
            self.img_sub = message_filters.Subscriber(self, Image, img_topic, qos_profile=qos_profile_sensor_data)
            self.pc_sub = message_filters.Subscriber(self, PointCloud2, pc_topic, qos_profile=qos_profile_sensor_data)
            
            self.ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.pc_sub], queue_size=10, slop=0.5)
            self.ts.registerCallback(self.sync_callback)

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS

        except Exception as e:
            self.get_logger().error(f"Failed to configure node: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating Body Pose Estimator Node...")
        super().on_activate(state)
        self._is_active = True
        self.latest_poses = [] # Clear any stale data on fresh activation
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating Body Pose Estimator Node...")
        self._is_active = False
        super().on_deactivate(state)
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up Body Pose Estimator resources...")
        
        self.destroy_publisher(self.marker_pub)
        self.destroy_publisher(self.debug_pub)
        
        # Destroy underlying ROS subscriptions in message_filters
        if self.img_sub is not None:
            self.destroy_subscription(self.img_sub.sub)
        if self.pc_sub is not None:
            self.destroy_subscription(self.pc_sub.sub)

        self.marker_pub = self.debug_pub = None
        self.img_sub = self.pc_sub = self.ts = None

        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down Body Pose Estimator Node...")
        if self.model is not None:
            self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # --- PROCESSING METHODS ---

    def get_xyz_from_cloud(self, cloud_msg, u, v, window=3):
        """Samples a window around (u,v) to avoid NaNs at edges."""
        points = []
        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                curr_u, curr_v = u + dx, v + dy
                if 0 <= curr_u < cloud_msg.width and 0 <= curr_v < cloud_msg.height:
                    offset = (curr_v * cloud_msg.row_step) + (curr_u * cloud_msg.point_step)
                    try:
                        x, y, z = struct.unpack_from('fff', cloud_msg.data, offset)
                        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                            points.append((x, y, z))
                    except: continue
        
        if not points:
            self.get_logger().warn(f"No valid depth found near pixel ({u}, {v})")
            return None
        return np.median(points, axis=0).tolist()

    def sync_callback(self, img_msg, pc_msg):
        if not self._is_active:
            return 

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        results = self.model(cv_image, conf=self.conf, verbose=False, retina_masks=True)
        if not results or results[0].masks is None:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
            return

        target_contour = self.find_body_contour(results[0])
        if target_contour is not None:
            four_points_approx = self.get_4_point_approx(target_contour)
            if four_points_approx is not None:
                sorted_points_2d = self.sort_points_clockwise(four_points_approx)

                points_3d = []
                for pt in sorted_points_2d:
                    u, v = pt
                    xyz = self.get_xyz_from_cloud(pc_msg, u, v, window=5)
                    if xyz: points_3d.append(xyz)
                
                if len(points_3d) == 4:
                    self.latest_poses = points_3d 
                    self.publish_results(points_3d, img_msg.header)
                    cv2.drawContours(cv_image, [sorted_points_2d.reshape(4,1,2)], -1, (0, 255, 0), 2)
                else:
                    self.get_logger().warn(f"3D extraction failed: {len(points_3d)}/4 points valid")

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

    def find_body_contour(self, result):
        body_id = next((i for i, name in result.names.items() if name == 'body'), None)
        if body_id is None and len(result.names) == 1:
            body_id = list(result.names.keys())[0]
        
        max_area, target_contour = 0, None
        if result.boxes is not None:
            for i, cls_tensor in enumerate(result.boxes.cls):
                if int(cls_tensor) == body_id:
                    contour = result.masks.xy[i].astype(np.int32).reshape(-1, 1, 2)
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area, target_contour = area, contour
        return target_contour

    def get_4_point_approx(self, contour):
        epsilon_factor = 0.01
        for _ in range(50):
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
            if len(approx) == 4: return approx
            if len(approx) < 4: return None
            epsilon_factor += 0.005
        return None
    
    def sort_points_clockwise(self, pts):
        # pts is (4, 1, 2) from approxPolyDP, reshape to (4, 2)
        pts = pts.reshape((4, 2))
        rect = np.zeros((4, 2), dtype="int32")

        # Top-left has the smallest sum, Bottom-right has the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-Left
        rect[2] = pts[np.argmax(s)] # Bottom-Right

        # Top-right has the smallest difference (x - y is large, or y - x is small)
        # Bottom-left has the largest difference (y - x is large)
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-Right
        rect[3] = pts[np.argmax(diff)] # Bottom-Left

        return rect

    def publish_results(self, points_3d, header):
        pts = np.array(points_3d)
        centroid = np.mean(pts, axis=0)
        
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered)
        R = vh.T
        if np.linalg.det(R) < 0: R[:, 2] *= -1

        try:
            from transforms3d.quaternions import mat2quat
            quat = mat2quat(R)
        except ImportError:
            quat = [1.0, 0.0, 0.0, 0.0]

        ma = MarkerArray()
        for i, pt in enumerate(points_3d):
            m = Marker(header=header, ns="corners", id=i, type=Marker.SPHERE, action=Marker.ADD)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pt)
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.a, m.color.r = 1.0, 1.0
            ma.markers.append(m)
        self.marker_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = BodyPoseEstimatorLifecycleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()