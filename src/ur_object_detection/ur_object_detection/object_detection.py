#!/usr/bin/env python3
import rclpy
from rclpy.node        import Node
from sensor_msgs.msg   import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
from ur_package_msgs.msg import ObjectPosition, ObjectPositions
from ur_package_msgs.srv import GetObjectLocations
from cv_bridge          import CvBridge
import image_geometry, cv2, numpy as np, torch
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops    import box_convert
import supervision as sv

# ─── CONFIG ──────────────────────────────────────────────────────────────
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD= 0.25
TEXT_PROMPT   = "white colored cube"

MODEL_CFG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WTS = "/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver"\
            "/ur_object_detection/ur_object_detection/weights/groundingdino_swint_ogc.pth"
CAPTURE_IMG = "/tmp/dino_capture.jpg"
# ──────────────────────────────────────────────────────────────────────────

class Deprojection(Node):
    def __init__(self):
        super().__init__("deprojection_node")
        self.depth_image   = None
        self.camera_info   = None
        self.color_image   = None
        self.latest_result = None

        self.bridge       = CvBridge()
        self.camera_model = image_geometry.PinholeCameraModel()

        # Load model
        self.model = load_model(MODEL_CFG, MODEL_WTS)
        self.get_logger().info("Grounding-DINO model loaded")

        # Subs
        self.create_subscription(Image,      "/camera/camera/depth/image_rect_raw",
                                 self.depth_cb, 10)
        self.create_subscription(CameraInfo, "/camera/camera/depth/camera_info",
                                 self.info_cb,  10)
        self.create_subscription(Image,      "/camera/camera/color/image_raw",
                                 self.rgb_cb,   10)

        # Service + publisher
        self.create_service(GetObjectLocations, "get_object_locations", self.srv_cb)
        self.pub = self.create_publisher(PoseStamped, "/orange_position", 10)

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")

    def info_cb(self, msg: CameraInfo):
        self.camera_info = msg
        self.camera_model.fromCameraInfo(msg)

    def rgb_cb(self, msg):
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def srv_cb(self, _req, resp):
        self.get_logger().info("📢 Service callback invoked")
        if self.color_image is None:
            self.get_logger().warn("No RGB image yet—nothing to do")
            return resp

        ok = cv2.imwrite(CAPTURE_IMG, cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        self.get_logger().info(f"Captured image write success? {ok}")

        # run DINO
        src, img = load_image(CAPTURE_IMG)
        boxes, logits, phrases = predict(
            self.model, img, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD
        )
        self.get_logger().info(f"DINO → {len(phrases)} phrases: {phrases}")

        result = ObjectPositions()
        h, w, _ = src.shape
        xyxy = box_convert(boxes * torch.Tensor([w, h, w, h]),
                           "cxcywh", "xyxy").numpy().astype(int)
        self.get_logger().info(f"Boxes array shape: {xyxy.shape}")

        any_published = False
        for i, phrase in enumerate(phrases):
            cx = int((xyxy[i,0] + xyxy[i,2]) / 2)
            cy = int((xyxy[i,1] + xyxy[i,3]) / 2)
            self.get_logger().info(f"→ Attempting pixel_to_cam at ({cx},{cy})")

            ps_cam = self.pixel_to_cam(cx, cy)
            if ps_cam is None:
                self.get_logger().warn(f"   pixel_to_cam returned None for ({cx},{cy})")
                continue

            self.get_logger().info(f"   pixel_to_cam → PoseStamped: {ps_cam.pose.position}")
            any_published = True
            self.pub.publish(ps_cam)
            self.get_logger().info(f"   Published on /orange_position")

            obj = ObjectPosition()
            obj.id            = int(i)
            obj.class_name    = phrase
            obj.pose          = ps_cam
            obj.x_min = int(xyxy[i,0])
            obj.y_min = int(xyxy[i,1])
            obj.x_max = int(xyxy[i,2])
            obj.y_max = int(xyxy[i,3])
            result.object_position.append(obj)

        if not any_published:
            self.get_logger().warn("⭕ No poses were published!")

        # attach annotated image
        annotated = self._annotate(src, boxes, logits, phrases)
        result.image = self.bridge.cv2_to_imgmsg(annotated, "bgr8")

        self.latest_result = result
        resp.result       = result
        return resp

    def _annotate(self, img, boxes, logits, phrases):
        h, w, _ = img.shape
        dets = sv.Detections(
            xyxy=box_convert(boxes * torch.Tensor([w,h,w,h]),
                             "cxcywh","xyxy").numpy()
        )
        labels = [f"{i}: {p} {l:.2f}" for i,(p,l) in enumerate(zip(phrases, logits))]
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(frame, dets)
        return sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX).annotate(frame, dets, labels)

    def pixel_to_cam(self, x, y):
        if self.depth_image is None or self.camera_info is None:
            return None

        patch = (self.depth_image[max(0,y-2):y+3, max(0,x-2):x+3]
                 .astype(np.float32) / 1000.0)
        valid = patch[(patch>0) & np.isfinite(patch)]
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
