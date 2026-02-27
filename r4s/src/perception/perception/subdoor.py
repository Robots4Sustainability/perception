import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import struct
import math
import os
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from std_srvs.srv import SetBool

def get_package_name_from_path(file_path):
    p = Path(file_path)
    try:
        idx = p.parts.index('site-packages')
        return p.parts[idx + 1]
    except (ValueError, IndexError):
        return "perception" # Fallback to your package name

class BodyPoseEstimatorNode(Node):
    def __init__(self):
        super().__init__('subdoor_pose_estimator')
        self.package_name = get_package_name_from_path(__file__)
        
        # --- Parameters ---
        self.declare_parameter('model_path', '') 
        self.declare_parameter('input_mode', 'robot') 
        self.declare_parameter('conf_threshold', 0.25)
        
        param_model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_mode = self.get_parameter('input_mode').get_parameter_value().string_value
        self.conf = self.get_parameter('conf_threshold').get_parameter_value().double_value
        
        # Resolve Model Path
        if param_model_path:
            self.model_path = param_model_path
        else:
            share_dir = get_package_share_directory(self.package_name)
            self.model_path = os.path.join(share_dir, 'models', 'subdoor.pt')

        # --- Setup YOLO ---
        self.get_logger().info(f"Loading YOLO model from: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        self.bridge = CvBridge()

        # --- Topics ---
        img_topic = '/camera/color/image_raw' if self.input_mode == 'robot' else '/camera/camera/color/image_raw'
        pc_topic = '/camera/depth/color/points' if self.input_mode == 'robot' else '/camera/camera/depth/color/points'

        # --- Synchronization ---
        self.img_sub = message_filters.Subscriber(self, Image, img_topic, qos_profile=qos_profile_sensor_data)
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, pc_topic, qos_profile=qos_profile_sensor_data)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([self.img_sub, self.pc_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.sync_callback)

        # --- Publishers ---
        self.marker_pub = self.create_publisher(MarkerArray, '/body_markers', 10)
        self.debug_pub = self.create_publisher(Image, '/body_debug_image', 10)

        self.is_active = False
        self.latest_poses = [] # Store the 4 corner points here
        
        # Create a Service to turn the processing on/off
        self.srv = self.create_service(SetBool, 'toggle_subdoor_estimation', self.toggle_callback)
        self.get_logger().info("Subdoor Worker initialized. Waiting for 'toggle_subdoor_estimation' service...")

    def toggle_callback(self, request, response):
        self.is_active = request.data
        self.latest_poses = [] # Clear previous results
        response.success = True
        response.message = f"Subdoor estimation {'started' if self.is_active else 'stopped'}"
        return response

    def get_xyz_from_cloud(self, cloud_msg, u, v, window=3):
        """Samples a window around (u,v) to avoid NaNs at edges."""
        points = []
        # Search in a small window
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
        # Return median to filter outliers/noise at the corner
        return np.median(points, axis=0).tolist()

    def sync_callback(self, img_msg, pc_msg):
        if not self.is_active:
            return # Do nothing if the pipeline hasn't woken us up

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
            four_points_2d = self.get_4_point_approx(target_contour)
            if four_points_2d is not None:
                points_3d = []
                for pt in four_points_2d:
                    u, v = pt[0]
                    xyz = self.get_xyz_from_cloud(pc_msg, u, v, window=5)
                    if xyz: points_3d.append(xyz)
                
                if len(points_3d) == 4:
                    self.latest_poses = points_3d 
                    self.publish_results(points_3d, img_msg.header)
                    cv2.drawContours(cv_image, [four_points_2d], -1, (0, 255, 0), 2)
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

    def publish_results(self, points_3d, header):
        pts = np.array(points_3d)
        centroid = np.mean(pts, axis=0)
        
        # Simple SVD for orientation
        centered = pts - centroid
        _, _, vh = np.linalg.svd(centered)
        R = vh.T
        if np.linalg.det(R) < 0: R[:, 2] *= -1

        try:
            from transforms3d.quaternions import mat2quat
            quat = mat2quat(R)
        except ImportError:
            quat = [1.0, 0.0, 0.0, 0.0]

        # Markers
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
    node = BodyPoseEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()