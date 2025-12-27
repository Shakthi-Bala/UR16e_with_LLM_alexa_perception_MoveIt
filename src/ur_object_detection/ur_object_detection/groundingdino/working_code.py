#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from ur_package_msgs.msg import ObjectPosition, ObjectPositions
from ur_package_msgs.srv import GetObjectLocations
from cv_bridge import CvBridge
import image_geometry
import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import supervision as sv
import tf2_ros
import tf2_geometry_msgs

# ─── Configuration ────────────────────────────────────────────────────────────
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
TEXT_PROMPT    = "white colored cube"

MODEL_CFG  = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WTS  = "/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver/ur_object_detection/ur_object_detection/weights/groundingdino_swint_ogc.pth"
IMAGE_PATH = "/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver/ur_object_detection/ur_object_detection/assets/generated_image.jpeg"
# ────────────────────────────────────────────────────────────────────────────────

class Deprojection(Node):
    def __init__(self):
        super().__init__("deprojection_node")

        # State
        self.depth_image  = None
        self.camera_info  = None
        self.color_image  = None
        self.latest_result = None
        self.recursion     = 0
        self.cv_bridge     = CvBridge()
        self.camera_model  = image_geometry.PinholeCameraModel()

        # Load GroundingDINO model
        self.model = load_model(MODEL_CFG, MODEL_WTS)
        self.get_logger().info("Loaded the GroundingDINO model")

        # TF2
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(
            Image, 
            "/camera/camera/depth/image_rect_raw", 
            self.depth_image_callback,  10
        )
        self.create_subscription(
            CameraInfo, 
            "/camera/camera/depth/camera_info", 
            self.camera_info_callback, 10
        )
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.color_image_callback, 10
        )

        # Service for on‐demand detection
        self.create_service(
            GetObjectLocations,
            "get_object_locations",
            self.handle_get_object_locations
        )

        # Publisher + Timer to stream first object pose
        self.stream_pub = self.create_publisher(PoseStamped, "/orange_position", 10)
        self.create_timer(0.5, self.publish_stream)

    # ─── Callbacks ─────────────────────────────────────────────────────────────
    def depth_image_callback(self, msg: Image):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info = msg
        self.camera_model.fromCameraInfo(msg)

    def color_image_callback(self, msg: Image):
        self.color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    # ─── Service handler ───────────────────────────────────────────────────────
    def handle_get_object_locations(self, request, response):
        # Save and prepare the RGB image
        rgb = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(IMAGE_PATH, rgb)

        # Run grounding DINO
        img_src, img = load_image(IMAGE_PATH)
        boxes, logits, phrases = predict(
            model=self.model,
            image=img,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # Annotate and store result image
        ann = self._annotate(img_src, boxes, logits, phrases)
        os.makedirs("inference_images", exist_ok=True)
        cv2.imwrite("inference_images/annotated_image.jpg", ann)

        # Build ObjectPositions message
        result_msg = ObjectPositions()
        h, w, _ = img_src.shape
        scaled = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)

        for i, phrase in enumerate(phrases):
            cx = int((xyxy[i][0] + xyxy[i][2]) / 2)
            cy = int((xyxy[i][1] + xyxy[i][3]) / 2)
            pose3d = self._get_3d_position(cx, cy)
            if pose3d is None:
                continue

            obj = ObjectPosition()
            obj.id       = i
            obj.class_name = phrase
            obj.pose     = pose3d
            obj.x_min    = xyxy[i][0]
            obj.y_min    = xyxy[i][1]
            obj.x_max    = xyxy[i][2]
            obj.y_max    = xyxy[i][3]
            result_msg.object_position.append(obj)

        # Attach annotated image & store for streaming
        result_msg.image      = self.cv_bridge.cv2_to_imgmsg(ann, encoding="bgr8")
        self.latest_result    = result_msg
        response.result       = result_msg
        return response

    # ─── Helpers ────────────────────────────────────────────────────────────────
    def _annotate(self, img_src, boxes, logits, phrases):
        h, w, _ = img_src.shape
        scaled = boxes * torch.Tensor([w, h, w, h])
        xyxy   = box_convert(scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        dets   = sv.Detections(xyxy=xyxy)
        labels = [f"{i}: {p} {l:.2f}" for i,(p,l) in enumerate(zip(phrases, logits))]

        b_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        l_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

        frame = cv2.cvtColor(img_src, cv2.COLOR_RGB2BGR)
        frame = b_annot.annotate(scene=frame, detections=dets)
        frame = l_annot.annotate(scene=frame, detections=dets, labels=labels)
        return frame

    def _get_3d_position(self, x, y):
        if self.depth_image is None or self.camera_info is None:
            return None

        # Depth window parameters
        window_size = 5
        half_size = window_size // 2
        valid_depths = []

        # Search neighborhood for valid depth
        for dy in range(-half_size, half_size + 1):
            for dx in range(-half_size, half_size + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.depth_image.shape[0] and 0 <= nx < self.depth_image.shape[1]:
                    d = float(self.depth_image[ny, nx]) / 1000.0  # mm → meters
                    if not np.isnan(d) and d > 0.1:  # Filter very small or NaN depths
                        valid_depths.append((d, (nx, ny)))

        if not valid_depths:
            return None

        # Use the closest valid depth point to center pixel
        valid_depths.sort(key=lambda item: np.hypot(item[1][0] - x, item[1][1] - y))
        best_depth, (best_x, best_y) = valid_depths[0]

        ray = self.camera_model.projectPixelTo3dRay((best_x, best_y))
        pt = np.array(ray) * best_depth

        ps = PoseStamped()
        ps.header.frame_id = self.camera_model.tf_frame
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose.position.x = float(pt[0])
        ps.pose.position.y = float(pt[1])
        ps.pose.position.z = float(pt[2])
        ps.pose.orientation.w = 1.0

        # Transform into "world" frame
        try:
            tf = self.tf_buffer.lookup_transform("world", ps.header.frame_id, Time())
            return tf2_geometry_msgs.do_transform_pose(ps, tf)
        except Exception as ex:
            self.get_logger().error(f"TF2 transform failed: {ex}")
            return None

    # Stream the first detected object's pose on a topic
    def publish_stream(self):
        if not self.latest_result or not self.latest_result.object_position:
            return
        first = self.latest_result.object_position[0].pose.pose
        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose            = first
        self.stream_pub.publish(ps)

def main(args=None):
    rclpy.init(args=args)
    node = Deprojection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
