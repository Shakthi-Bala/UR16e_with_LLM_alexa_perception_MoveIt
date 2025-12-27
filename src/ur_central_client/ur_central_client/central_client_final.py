#!/usr/bin/env python3
import base64
import json
import time

import cv2
import rclpy
from openai import OpenAI
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from ur_package_msgs.action import PickAction
from ur_package_msgs.srv import GetObjectLocations, GetSpeech
from cv_bridge import CvBridge


class CentralClient(Node):
    def __init__(self):
        super().__init__('central_client')

        # Perception service client
        self.perception_cli = self.create_client(GetObjectLocations, 'get_object_locations')
        self.get_logger().info('Waiting for perception service...')
        if not self.perception_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Perception service not available!')

        # Speech-to-text service client
        self.speech_cli = self.create_client(GetSpeech, 'get_speech')
        self.get_logger().info('Waiting for speech service...')
        if not self.speech_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Speech service not available!')

        # Pick action client
        self.pick_action_cli = ActionClient(self, PickAction, 'pick_action')
        self.get_logger().info('Waiting for pick_action server...')
        if not self.pick_action_cli.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('PickAction server not available!')

        # LLM client
        self.llm = OpenAI()

        # CV bridge
        self.bridge = CvBridge()

    def call_perception(self):
        req = GetObjectLocations.Request()
        future = self.perception_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('Perception call failed')
            return None
        return future.result().result

    def call_speech(self):
        """Invoke your ROS service that returns the captured speech as a string."""
        req = GetSpeech.Request()
        future = self.speech_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('Speech call failed')
            return ''
        prompt = future.result().captured_speech
        self.get_logger().info(f'Received speech prompt: "{prompt}"')
        return prompt

    def choose_objects_with_llm(self, prompt, detections, image):
        _, buf = cv2.imencode('.jpg', image)
        img_b64 = base64.b64encode(buf).decode('utf-8')

        det_list = [{'id': o.id, 'label': o.Class} for o in detections]
        det_json = json.dumps(det_list)

        system = 'You pick objects by ID; respond with a JSON list of IDs, e.g. [0,2].'
        resp = self.llm.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role':'system','content':system},
                {'role':'user','content':f'Prompt: {prompt}\nDetections: {det_json}'},
                {'role':'user','content':f'data:image/jpeg;base64,{img_b64}'},
            ],
            temperature=0.0
        )

        content = resp.choices[0].message.content.strip()
        try:
            picks = json.loads(content)
            return [int(i) for i in picks]
        except Exception as e:
            self.get_logger().error(f'LLM parse error: {e}')
            return []

    def send_pick(self, pose: PoseStamped):
        goal = PickAction.Goal()
        goal.source = pose
        send_future = self.pick_action_cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if not gh.accepted:
            self.get_logger().error('PickAction goal rejected')
            return False
        res_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_future)
        result = res_future.result().result
        self.get_logger().info(f'PickAction result: {result.result}')
        return result.result

    def run(self):
        # 1) Get the spoken prompt instead of typing
        prompt = self.call_speech()
        if not prompt:
            return

        # 2) Perception
        per = self.call_perception()
        if not per or not per.object_position:
            self.get_logger().warn('No objects detected')
            return

        img = self.bridge.imgmsg_to_cv2(per.image, 'bgr8')
        cv2.imwrite('annotated.png', img)

        # 3) LLM decides which IDs
        pick_ids = self.choose_objects_with_llm(prompt, per.object_position, img)
        if not pick_ids:
            self.get_logger().warn('LLM returned no picks')
            return

        # 4) Send pick goals
        for idx in pick_ids:
            if idx < 0 or idx >= len(per.object_position):
                self.get_logger().warn(f'Invalid ID {idx}')
                continue
            pose = per.object_position[idx].pose
            if not self.send_pick(pose):
                break

def main(args=None):
    rclpy.init(args=args)
    node = CentralClient()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
