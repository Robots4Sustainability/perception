import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from std_msgs.msg import Float32 # <--- NEW: Import for the radius topic
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
import cv2
import torch
from ultralytics import YOLO, SAM

class ObjectSphereNode(Node):
    def __init__(self):
        super().__init__('object_sphere_node')

        # --- CONFIGURATION ---
        self.img_topic = '/camera/color/image_raw'
        self.pc_topic = '/camera/depth/color/points'
        self.yolo_model = '/home/mohsin/fix/r4s/models/fine_tuned.pt'
        self.sam_model = "sam_b.pt"
        self.target_classes = ["unit", "motor"]
        # Define BGR colors for classes (Green for unit, Blue for motor)
        self.class_colors = {"unit": (0, 255, 0), "motor": (255, 0, 0)}
        self.min_points_threshold = 50 # Minimum valid points needed for 3D calculation

        # --- MODELS ---
        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Loading Models on {self.device}...")
        self.yolo = YOLO(self.yolo_model).to(self.device)
        self.sam = SAM(self.sam_model).to(self.device)

        # --- PUBLISHERS ---
        # 1. 3D Markers for RViz Visualization
        self.marker_pub = self.create_publisher(MarkerArray, '/object_markers', 10)
        # 2. 3D Data for other nodes (position/size)
        self.detect_pub = self.create_publisher(Detection3DArray, '/object_detections', 10)
        # 3. 2D Segmentation visualization image
        self.seg_pub = self.create_publisher(Image, '/object_segmentation', 10)
        # 4. NEW: Single float radius publisher for external node
        self.radius_pub = self.create_publisher(Float32, '/perception/detected_object_radius', 10)

        # --- SUBSCRIBERS ---
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = message_filters.Subscriber(self, Image, self.img_topic, qos_profile=qos)
        self.pc_sub = message_filters.Subscriber(self, PointCloud2, self.pc_topic, qos_profile=qos)
        
        # Sync RGB and Depth
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.img_sub, self.pc_sub], queue_size=5, slop=0.1
        )
        self.ts.registerCallback(self.callback)
        self.get_logger().info("Node ready.")

    def callback(self, img_msg, pc_msg):
        # 1. Image Conversion
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge Error: {e}")
            return

        # 2. YOLO Inference
        yolo_res = self.yolo(cv_img, verbose=False, conf=0.5)[0]
        boxes, labels = [], []
        for box in yolo_res.boxes:
            cls_id = int(box.cls[0])
            name = yolo_res.names[cls_id]
            if name in self.target_classes:
                boxes.append(box.xyxy[0].cpu().numpy())
                labels.append(name)
        
        if not boxes: return

        # 3. SAM Inference
        boxes_np = np.array(boxes)
        sam_res = self.sam(cv_img, bboxes=boxes_np, verbose=False)[0]
        if sam_res.masks is None: return
        masks_data = sam_res.masks.data.cpu().numpy()

        # 4. Point Cloud Prep
        if pc_msg.height <= 1: return
        points = pc2.read_points_numpy(pc_msg, field_names=("x", "y", "z"), skip_nans=False)
        try:
            points_3d = points.reshape((pc_msg.height, pc_msg.width, 3))
        except ValueError: return

        # --- Initialize Outputs ---
        marker_array = MarkerArray()
        detection_array = Detection3DArray()
        detection_array.header = pc_msg.header
        
        # Create a black background image for segmentation visualization
        seg_overlay_img = np.zeros((cv_img.shape[0], cv_img.shape[1], 3), dtype=np.uint8)
        
        # NEW: Flag to track if the radius has been published in this frame
        radius_published = False 

        # 5. Iterate Results
        for i, (mask_small, label) in enumerate(zip(masks_data, labels)):
            # A. Resize Mask
            mask_uint8 = mask_small.astype(np.uint8)
            # Resize to match the original image/pointcloud dimensions
            mask_resized = cv2.resize(mask_uint8, (pc_msg.width, pc_msg.height), interpolation=cv2.INTER_NEAREST)
            full_mask_bool = mask_resized.astype(bool)

            # --- Paint Segmentation Image ---
            # Get BGR color for this class
            color_bgr = self.class_colors.get(label, (255, 255, 255))
            # Use boolean indexing to color the pixels where the mask is True
            seg_overlay_img[full_mask_bool] = color_bgr

            # B. Extract 3D Points based on mask
            obj_points = points_3d[full_mask_bool]
            valid_obj_points = obj_points[~np.isnan(obj_points).any(axis=1)]
            
            if len(valid_obj_points) < self.min_points_threshold: continue 

            # C. Calculate Sphere
            centroid = np.mean(valid_obj_points, axis=0)
            distances = np.linalg.norm(valid_obj_points - centroid, axis=1)
            radius = np.max(distances)
            diameter = radius * 2.0
            
            # D. NEW: Publish radius for the first valid object
            if not radius_published:
                radius_msg = Float32()
                radius_msg.data = float(radius)
                self.radius_pub.publish(radius_msg)
                self.get_logger().debug(f"Published radius for {label}: {radius:.4f} m")
                radius_published = True

            # E. Create Marker (RViz)
            sphere = Marker()
            sphere.header = pc_msg.header
            sphere.ns = "detections"
            sphere.id = i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(centroid[0])
            sphere.pose.position.y = float(centroid[1])
            sphere.pose.position.z = float(centroid[2])
            sphere.scale.x = float(diameter); sphere.scale.y = float(diameter); sphere.scale.z = float(diameter)
            # Set color based on class, matching segmentation color but semi-transparent
            sphere.color.r = color_bgr[2] / 255.0 # BGR to RGB normalized
            sphere.color.g = color_bgr[1] / 255.0
            sphere.color.b = color_bgr[0] / 255.0
            sphere.color.a = 0.5
            marker_array.markers.append(sphere)

            # F. Create Detection Message (Data)
            detection = Detection3D()
            detection.header = pc_msg.header
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = label
            hypothesis.hypothesis.score = 1.0 
            detection.results.append(hypothesis)
            detection.bbox.center.position.x = float(centroid[0])
            detection.bbox.center.position.y = float(centroid[1])
            detection.bbox.center.position.z = float(centroid[2])
            detection.bbox.size.x = float(diameter); detection.bbox.size.y = float(diameter); detection.bbox.size.z = float(diameter)
            detection_array.detections.append(detection)

        # --- Publish All outputs ---
        self.marker_pub.publish(marker_array)
        self.detect_pub.publish(detection_array)

        # Publish Segmentation Image
        try:
            # Convert numpy array back to ROS Image message
            seg_msg = self.bridge.cv2_to_imgmsg(seg_overlay_img, encoding="bgr8")
            # IMPORTANT: Copy header from input to ensure sync in RViz
            seg_msg.header = img_msg.header 
            self.seg_pub.publish(seg_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish segmentation image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectSphereNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()