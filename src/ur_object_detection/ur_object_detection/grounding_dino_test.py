from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os

model = load_model("/home/alien/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver/ur_object_detection/ur_object_detection/weights/groundingdino_swint_ogc.pth")
IMAGE_PATH = "/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver/ur_object_detection/ur_object_detection/assets/test1.png"
TEXT_PROMPT = "detect .blue_cube.red_cube.table"
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# image = cv2.imread(IMAGE_PATH)

image_source, image = load_image(IMAGE_PATH)

boxes, logits   , phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

print("########################")
print("\nBoxes : ")
print(boxes, type(boxes))
print("\n")

print("########################")
print("\nLogits : ")
print(logits, type(logits))
print("\n")

print("########################")
print("\nPhrases : ")
print(phrases, type(phrases))
print("\n")

cv2.imwrite("/home/alien/workspace/ros_ur_driver/src/Universal_Robots_ROS2_Driver/ur_object_detection/ur_object_detection/inference_images/annotated_image.jpeg", annotated_frame)