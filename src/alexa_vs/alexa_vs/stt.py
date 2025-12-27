#!/usr/bin/env python3
from flask import Flask
from ask_sdk_core.skill_builder import SkillBuilder
from flask_ask_sdk.skill_adapter import SkillAdapter
from ask_sdk_core.dispatch_components import AbstractRequestHandler, AbstractExceptionHandler
from ask_sdk_core.utils import is_request_type, is_intent_name
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_model.ui import SimpleCard
import rclpy
from rclpy.node import Node
import threading
from ur_package_msgs.srv import GetSpeech  

app = Flask(__name__)
latest_speech = "No speech captured yet."

# ---------------- ROS 2 Service Node ----------------
class SpeechService(Node):
    def __init__(self):
        super().__init__('speech_service')
        self.srv = self.create_service(GetSpeech, 'get_speech', self.get_speech_callback)
    
    def get_speech_callback(self, request, response):
        global latest_speech
        response.captured_speech = latest_speech
        self.get_logger().info(f"Returning speech: {latest_speech}")
        return response

# ---------------- Alexa Skill Handlers ----------------
class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speech_text = "Hi, how can we help?"
        handler_input.response_builder.speak(speech_text).set_card(
            SimpleCard("Hello World", speech_text)).set_should_end_session(False)
        return handler_input.response_builder.response
    

class CaptureSpeechIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return is_intent_name("CaptureSpeechIntent")(handler_input)

    def handle(self, handler_input):
        global latest_speech

        if "speech" in handler_input.request_envelope.request.intent.slots:
            captured_speech = handler_input.request_envelope.request.intent.slots["speech"].value
        else:
            captured_speech = "No speech captured."

        latest_speech = captured_speech  

        print(f"Captured Speech: {captured_speech}")

        speech_text = f"You said: {captured_speech}"
        handler_input.response_builder.speak(speech_text).set_card(
            SimpleCard("Speech Capture", speech_text)).set_should_end_session(False)

        return handler_input.response_builder.response
    
class AllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        speech = "Sorry, I couldn't understand that. Can you repeat?"
        handler_input.response_builder.speak(speech).ask(speech)
        return handler_input.response_builder.response

# ---------------- Skill Adapter Setup ----------------
skill_builder = SkillBuilder()
skill_builder.add_request_handler(LaunchRequestHandler())
skill_builder.add_request_handler(CaptureSpeechIntentHandler())
skill_builder.add_exception_handler(AllExceptionHandler())

skill_adapter = SkillAdapter(
    skill=skill_builder.create(), 
    skill_id="amzn1.ask.skill.54b0c020-dbd2-4d40-9d0f-61819c059b42",
    app=app)

@app.route("/")
def invoke_skill():
    return skill_adapter.dispatch_request()

skill_adapter.register(app=app, route="/")

# ---------------- ROS Thread Spin ----------------
def ros_spin():
    rclpy.init()
    node = SpeechService()
    rclpy.spin(node)
    rclpy.shutdown()

# ---------------- Main ----------------
if __name__ == '__main__':
    threading.Thread(target=ros_spin, daemon=True).start()
    app.run()
