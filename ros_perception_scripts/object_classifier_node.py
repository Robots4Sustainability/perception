import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray

class ObjectClassifierNode(Node):
    """
    A ROS2 node that subscribes to detections from a YOLO node, classifies them
    as pickable or non-pickable, and publishes them to separate topics.
    """
    def __init__(self):
        super().__init__('object_classifier_node')

        # Define the lists of pickable and non-pickable object classes
        self.pickable_classes = ['motor', 'unit']
        self.non_pickable_classes = ['wire_plugs', 'enclosure', 'speaker', 'assembly']
        
        self.get_logger().info(f"Pickable objects: {self.pickable_classes}")
        self.get_logger().info(f"Non-pickable objects: {self.non_pickable_classes}")

        # Create a subscriber to the main detections topic
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            10)

        # Create publishers for the classified objects
        self.pickable_pub = self.create_publisher(
            Detection2DArray, 
            '/pickable_objects', 
            10)
            
        self.non_pickable_pub = self.create_publisher(
            Detection2DArray, 
            '/non_pickable_objects', 
            10)

        self.get_logger().info('Object Classifier Node has started.')

    def detection_callback(self, msg: Detection2DArray):
        """
        Callback function to process incoming detections.
        """
        # Create new Detection2DArray messages for each category
        pickable_detections_msg = Detection2DArray()
        pickable_detections_msg.header = msg.header

        non_pickable_detections_msg = Detection2DArray()
        non_pickable_detections_msg.header = msg.header

        # Iterate through all detections in the received message
        for detection in msg.detections:
            # The class_id is typically in the first hypothesis result
            if not detection.results:
                continue

            class_id = detection.results[0].hypothesis.class_id

            # Check which list the class_id belongs to and append
            if class_id in self.pickable_classes:
                pickable_detections_msg.detections.append(detection)
            elif class_id in self.non_pickable_classes:
                non_pickable_detections_msg.detections.append(detection)
            else:
                self.get_logger().warn(f"Detected object '{class_id}' is not in any classification list.")

        # Publish the sorted lists, only if they contain detections
        if pickable_detections_msg.detections:
            self.pickable_pub.publish(pickable_detections_msg)

        if non_pickable_detections_msg.detections:
            self.non_pickable_pub.publish(non_pickable_detections_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectClassifierNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()