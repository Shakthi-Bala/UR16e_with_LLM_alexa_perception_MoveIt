#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg   import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from ur_package_msgs.msg import ObjectPosition, ObjectPositions
from ur_package_msgs.srv import GetObjectLocations
from cv_bridge import CvBridge

import numpy as np
import image_geometry, cv2, torch
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import supervision as sv

# ─── quaternion helpers ───────────────────────────────────────────────────────
def quat_inv(q):
    # q = [x,y,z,w]
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)

def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y - z*w),  2*(x*z + y*w)],
        [2*(x*y + z*w),  1-2*(x*x+z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),  2*(y*z + x*w),  1-2*(x*x+y*y)]
    ], dtype=float)
# ────────────────────────────────────────────────────────────────────────────────

# ─── CONFIG ───────────────────────────────────────────────────────────────────
BOX_THRESHOLD  = 0.35
TEXT_THRESHOLD = 0.25
TEXT_PROMPT    = "white colored cube"

MODEL_CFG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WTS = (
    "/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver"
    "/ur_object_detection/ur_object_detection/weights/groundingdino_swint_ogc.pth"
)
CAPTURE_IMG = "/tmp/dino_capture.jpg"
# ────────────────────────────────────────────────────────────────────────────────

class Deprojection(Node):
    def __init__(self):
        super().__init__("deprojection_node")

        # camera model & images
        self.bridge       = CvBridge()
        self.camera_model = image_geometry.PinholeCameraModel()
        self.depth_image  = None
        self.camera_info  = None
        self.color_image  = None

        # load DINO
        self.model = load_model(MODEL_CFG, MODEL_WTS)
        self.get_logger().info("Grounding-DINO model loaded")

        # subscribers
        self.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",
            self.depth_cb, 10
        )
        self.create_subscription(
            CameraInfo,
            "/camera/camera/depth/camera_info",
            self.info_cb, 10
        )
        self.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self.rgb_cb, 10
        )

        # service + publisher
        self.create_service(
            GetObjectLocations,
            "get_object_locations",
            self.srv_cb
        )
        self.pub = self.create_publisher(
            PoseStamped,
            "/orange_position_base",
            10
        )

        # ── Precompute your transforms ────────────────────────────────────────
        # world → camera_optical_frame
        self.q_wc = np.array([-0.213, -0.001, 0.977, -0.016], dtype=float)
        self.t_wc = np.array([0.673, 0.191, 0.322], dtype=float)
        # world → base_link
        self.q_wb = np.array([0.005, -0.002, 0.718, 0.696], dtype=float)
        self.t_wb = np.array([0.146, 0.596, 0.010], dtype=float)

        # rotation matrices
        self.R_wc = quat_to_mat(self.q_wc)
        self.R_bw = quat_to_mat(quat_inv(self.q_wb))

    def depth_cb(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg
        self.camera_model.fromCameraInfo(msg)

    def rgb_cb(self, msg: Image):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def srv_cb(self, _req, resp):
        self.get_logger().info("📢 Service callback invoked")
        if self.color_image is None:
            self.get_logger().warn("No RGB image yet—nothing to do")
            return resp

        # run DINO
        cv2.imwrite(
            CAPTURE_IMG,
            cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        )
        src, img = load_image(CAPTURE_IMG)
        boxes, logits, phrases = predict(
            self.model, img,
            TEXT_PROMPT,
            BOX_THRESHOLD,
            TEXT_THRESHOLD
        )
        self.get_logger().info(f"DINO → {len(phrases)} phrases: {phrases}")

        result = ObjectPositions()
        h, w, _ = src.shape
        xyxy = box_convert(
            boxes * torch.Tensor([w, h, w, h]),
            "cxcywh", "xyxy"
        ).numpy().astype(int)

        any_pub = False
        for i, phrase in enumerate(phrases):
            # pixel center
            cx = (xyxy[i,0] + xyxy[i,2]) // 2
            cy = (xyxy[i,1] + xyxy[i,3]) // 2

            ps_cam = self.pixel_to_cam(cx, cy)
            if ps_cam is None:
                continue

            # camera-frame point
            p_c = np.array([
                ps_cam.pose.position.x,
                ps_cam.pose.position.y,
                ps_cam.pose.position.z
            ], dtype=float)
            self.get_logger().info(
                f" camera_frame → x={p_c[0]:.3f}, y={p_c[1]:.3f}, z={p_c[2]:.3f}"
            )

            # world-frame: R_wc * p_c + t_wc
            p_w = self.R_wc.dot(p_c) + self.t_wc
            # base-frame: R_bw * (p_w - t_wb)
            p_b = self.R_bw.dot(p_w - self.t_wb)
            self.get_logger().info(
                f" base_link → x={p_b[0]:.3f}, y={p_b[1]:.3f}, z={p_b[2]:.3f}"
            )

            # build PoseStamped in base_link
            ps_base = PoseStamped()
            ps_base.header.frame_id = "base_link"
            ps_base.header.stamp    = ps_cam.header.stamp
            ps_base.pose.position.x = float(p_b[0])
            ps_base.pose.position.y = float(p_b[1])
            ps_base.pose.position.z = float(p_b[2])
            # copy orientation from camera (or leave identity)
            ps_base.pose.orientation = ps_cam.pose.orientation

            # publish & record
            self.pub.publish(ps_base)
            obj = ObjectPosition(
                id=int(i),
                class_name=phrase,
                pose=ps_base,
                x_min=int(xyxy[i,0]),
                x_max=int(xyxy[i,2]),
                y_min=int(xyxy[i,1]),
                y_max=int(xyxy[i,3]),
            )
            result.object_position.append(obj)
            any_pub = True

        if not any_pub:
            self.get_logger().warn("⭕ No poses were published!")

        # annotate
        dets = sv.Detections(xyxy=xyxy.astype(float))
        labels = [
            f"{j}: {p} {l:.2f}"
            for j, (p, l) in enumerate(zip(phrases, logits))
        ]
        frame = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
        frame = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(frame, dets)
        frame = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(frame, dets, labels)
        result.image = self.bridge.cv2_to_imgmsg(frame, "bgr8")

        # **Ensure we return the response**
        resp.result = result
        return resp

    def pixel_to_cam(self, x, y):
        if self.depth_image is None or self.camera_info is None:
            return None

        patch = (
            self.depth_image[max(0, y-2):y+3, max(0, x-2):x+3]
            .astype(np.float32) / 1000.0
        )
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size == 0:
            return None

        depth = float(np.median(valid))
        ray   = self.camera_model.projectPixelTo3dRay((x, y))
        pt    = np.array(ray) * depth

        ps = PoseStamped()
        ps.header.frame_id = self.camera_model.tf_frame
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pt)
        ps.pose.orientation.w = 1.0
        return ps

def main():
    rclpy.init()
    node = Deprojection()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
