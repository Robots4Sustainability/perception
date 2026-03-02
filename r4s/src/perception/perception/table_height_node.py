import rclpy
from rclpy.lifecycle import Node as LifecycleNode, State, TransitionCallbackReturn
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
import cv2
import torch
from ultralytics import YOLOWorld, SAM
from sklearn.linear_model import RANSACRegressor
from std_msgs.msg import Float32

class TableHeightNode(LifecycleNode):
    def __init__(self):
        super().__init__('table_height_estimator')

        # --- CONFIGURATION (Set in Unconfigured State) ---
        self.img_topic = '/camera/color/image_raw'
        self.pc_topic = '/camera/depth/color/points'
        self.custom_classes = ["white standing desk", "white table surface"]
        self.conf_threshold = 0.2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Placeholders for resources
        self.det_model = None
        self.seg_model = None
        self.bridge = CvBridge()
        
        self.marker_pub = None
        self.debug_pc_pub = None
        self.seg_img_pub = None
        self.height_pub = None
        
        self.img_sub = None
        self.pc_sub = None
        self.ts = None

        self._is_active = False
        self.get_logger().info("Table Height Lifecycle Node Initialized (Unconfigured).")

    # --- LIFECYCLE CALLBACKS ---

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info(f"Configuring: Loading models on {self.device}...")
        
        try:
            # 1. Load ML Models
            self.det_model = YOLOWorld('yolov8l-worldv2.pt')
            self.det_model.set_classes(self.custom_classes)
            self.det_model.to(self.device)
            
            self.seg_model = SAM('sam_b.pt')
            self.seg_model.to(self.device)

            # 2. Create Lifecycle Publishers
            self.marker_pub = self.create_lifecycle_publisher(MarkerArray, '/table_height_visualization', 10)
            self.debug_pc_pub = self.create_lifecycle_publisher(PointCloud2, '/table_points_debug', 10)
            self.seg_img_pub = self.create_lifecycle_publisher(Image, '/table_segmentation_image', 10)
            self.height_pub = self.create_lifecycle_publisher(Float32, '/table_height_value', 10)

            # 3. Create Subscribers & Synchronizer
            self.img_sub = message_filters.Subscriber(self, Image, self.img_topic)
            self.pc_sub = message_filters.Subscriber(self, PointCloud2, self.pc_topic)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.img_sub, self.pc_sub], queue_size=10, slop=0.1
            )
            self.ts.registerCallback(self.callback)

            self.get_logger().info("Configuration complete.")
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure: {e}")
            return TransitionCallbackReturn.FAILURE

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating node...")
        # super().on_activate() is required to activate the lifecycle publishers
        super().on_activate(state)
        self._is_active = True
        self.get_logger().info("Node activated. Processing incoming data.")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating node...")
        self._is_active = False
        # super().on_deactivate() deactivates the lifecycle publishers
        super().on_deactivate(state)
        self.get_logger().info("Node deactivated. Pausing processing.")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Cleaning up resources...")
        
        # 1. Destroy Subscriptions and Publishers
        self.destroy_publisher(self.marker_pub)
        self.destroy_publisher(self.debug_pc_pub)
        self.destroy_publisher(self.seg_img_pub)
        self.destroy_publisher(self.height_pub)
        
        # message_filters.Subscriber doesn't have a direct destroy method, 
        # so we destroy the underlying ROS subscription.
        if self.img_sub is not None:
            self.destroy_subscription(self.img_sub.sub)
        if self.pc_sub is not None:
            self.destroy_subscription(self.pc_sub.sub)

        self.marker_pub = self.debug_pc_pub = self.seg_img_pub = None
        self.img_sub = self.pc_sub = self.ts = None

        # 2. Unload Models and Free GPU Memory
        self.det_model = None
        self.seg_model = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        self.get_logger().info("Cleanup complete. GPU memory freed.")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Shutting down node...")
        # If shutting down from active/configured, ensure cleanup happens
        if self.det_model is not None:
            self.on_cleanup(state)
        return TransitionCallbackReturn.SUCCESS

    # --- PROCESSING CALLBACK ---

    def callback(self, img_msg, pc_msg):
        # Guard clause: Do nothing if the node is not active
        if not self._is_active:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # 1. YOLO-World (Find Table)
        det_results = self.det_model.predict(cv_image, conf=self.conf_threshold, verbose=False)
        bboxes = det_results[0].boxes.xyxy.tolist()
        if not bboxes: return

        # 2. SAM (Segment Table)
        table_box = [bboxes[0]] 
        seg_results = self.seg_model(cv_image, bboxes=table_box, verbose=False)
        if seg_results[0].masks is None: return
        table_mask = seg_results[0].masks.data[0].cpu().numpy()

        # 3. Visualization Image
        self.publish_debug_image(cv_image, table_mask, bboxes[0], img_msg.header)

        # 4. Calculate Height Logic
        self.process_point_cloud(pc_msg, table_mask)

    def process_point_cloud(self, pc_msg, table_mask):
        if pc_msg.height <= 1: return
        
        # --- A. Parse PointCloud into (H, W, 3) Array ---
        raw_data = np.frombuffer(pc_msg.data, dtype=np.uint8)
        try:
            raw_data = raw_data.reshape(pc_msg.height, pc_msg.row_step)
        except ValueError: return
        
        bytes_per_pixel = pc_msg.point_step
        raw_data = raw_data[:, :pc_msg.width * bytes_per_pixel]
        pixel_chunks = raw_data.reshape(pc_msg.height, pc_msg.width, bytes_per_pixel)

        off_x, off_y, off_z = 0, 4, 8 
        for field in pc_msg.fields:
            if field.name == 'x': off_x = field.offset
            if field.name == 'y': off_y = field.offset
            if field.name == 'z': off_z = field.offset

        x = pixel_chunks[:, :, off_x : off_x+4].view(dtype=np.float32).squeeze()
        y = pixel_chunks[:, :, off_y : off_y+4].view(dtype=np.float32).squeeze()
        z = pixel_chunks[:, :, off_z : off_z+4].view(dtype=np.float32).squeeze()
        
        points_3d = np.dstack((x, y, z))

        # --- B. Get Table Center ---
        table_pts = points_3d[table_mask]
        valid_table = table_pts[~np.isnan(table_pts).any(axis=1)]
        if valid_table.shape[0] < 50: return

        t_x = np.median(valid_table[:, 0])
        t_y = np.median(valid_table[:, 1])
        t_z = np.median(valid_table[:, 2])

        # --- C. Get Floor Position (RANSAC) ---
        h, w, _ = points_3d.shape
        floor_region_mask = np.zeros((h, w), dtype=bool)
        floor_region_mask[int(h*0.5):, :] = True
        
        floor_candidates_mask = floor_region_mask & (~table_mask)
        floor_pts = points_3d[floor_candidates_mask][::10]
        valid_floor = floor_pts[~np.isnan(floor_pts).any(axis=1)]

        floor_y_at_table = None

        if valid_floor.shape[0] > 100:
            X_in = valid_floor[:, [0, 2]] 
            Y_out = valid_floor[:, 1]    

            ransac = RANSACRegressor(residual_threshold=0.05)
            try:
                ransac.fit(X_in, Y_out)
                floor_y_at_table = ransac.predict([[t_x, t_z]])[0]
                if floor_y_at_table < t_y + 0.1: 
                    floor_y_at_table = None
            except:
                pass

        # --- D. Visualize ---
        self.publish_markers(t_x, t_y, t_z, floor_y_at_table, pc_msg.header)
        
        debug_cloud = pc2.create_cloud_xyz32(pc_msg.header, valid_table)
        self.debug_pc_pub.publish(debug_cloud)

    def publish_markers(self, tx, ty, tz, fy, header):
        marker_array = MarkerArray()

        m_table = Marker()
        m_table.header = header
        m_table.ns = "table"
        m_table.id = 0
        m_table.type = Marker.SPHERE
        m_table.action = Marker.ADD
        m_table.pose.position.x, m_table.pose.position.y, m_table.pose.position.z = float(tx), float(ty), float(tz)
        m_table.scale.x = m_table.scale.y = m_table.scale.z = 0.08
        m_table.color.r, m_table.color.g, m_table.color.b, m_table.color.a = 1.0, 0.0, 0.0, 1.0
        marker_array.markers.append(m_table)

        log_msg = f"Table Z: {tz:.2f}m"

        if fy is not None:
            height_meters = abs(fy - ty)
            log_msg += f" | Floor Est: {fy:.2f}m | HEIGHT: {height_meters:.3f}m"

            height_msg = Float32()
            height_msg.data = float(height_meters)
            self.height_pub.publish(height_msg)

            m_floor = Marker()
            m_floor.header = header
            m_floor.ns = "floor"
            m_floor.id = 1
            m_floor.type = Marker.CUBE
            m_floor.action = Marker.ADD
            m_floor.pose.position.x, m_floor.pose.position.y, m_floor.pose.position.z = float(tx), float(fy), float(tz)
            m_floor.scale.x, m_floor.scale.z = 0.2, 0.2
            m_floor.scale.y = 0.005 
            m_floor.color.r, m_floor.color.g, m_floor.color.b, m_floor.color.a = 0.0, 1.0, 0.0, 1.0
            marker_array.markers.append(m_floor)

            m_line = Marker()
            m_line.header = header
            m_line.ns = "line"
            m_line.id = 2
            m_line.type = Marker.LINE_LIST
            m_line.action = Marker.ADD
            m_line.scale.x = 0.005 
            m_line.color.r, m_line.color.g, m_line.color.b, m_line.color.a = 1.0, 1.0, 0.0, 1.0
            m_line.points.append(m_table.pose.position)
            m_line.points.append(m_floor.pose.position)
            marker_array.markers.append(m_line)

            m_text = Marker()
            m_text.header = header
            m_text.ns = "text"
            m_text.id = 3
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.text = f"{height_meters:.2f}m"
            m_text.pose.position.x = float(tx) + 0.15
            m_text.pose.position.y = (ty + fy) / 2.0  
            m_text.pose.position.z = float(tz)
            m_text.scale.z = 0.05 
            m_text.color.r, m_text.color.g, m_text.color.b, m_text.color.a = 1.0, 1.0, 1.0, 1.0
            marker_array.markers.append(m_text)

        self.get_logger().info(log_msg)
        self.marker_pub.publish(marker_array)

    def publish_debug_image(self, cv_image, mask, bbox, header):
        overlay = cv_image.copy()
        overlay[mask] = [0, 255, 0]
        blended = cv2.addWeighted(overlay, 0.4, cv_image, 0.6, 0)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0,0,255), 2)
        
        msg = self.bridge.cv2_to_imgmsg(blended, encoding="bgr8")
        msg.header = header
        self.seg_img_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args) 
    node = TableHeightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()