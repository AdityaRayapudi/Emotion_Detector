import eventlet
eventlet.monkey_patch()


from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import base64
import os
import numpy as np
import cv2
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

from EmotionDetection import NeuralNetwork, LabelMap

#Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode="eventlet")


# Initalize pytorch model
model = NeuralNetwork()
model.load_state_dict(torch.load("base_emotion_model.pth", weights_only=True))

# Initalize cv2 face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
convert_tensor = transforms.ToTensor()

def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def gen_frames(frame):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face = faceCascade.detectMultiScale(
                gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )

            # Draw Bouding Box on Face and Emotion Label
            for (x, y, w, h) in face:
                # Draw a green box around the faces
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

                # Get a gray scale image of of the face and down scale to 48x48
                gray_face = gray_frame[y:y + h, x:x + w]
                zoomed_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_LINEAR)

                # Convert image to Tensor
                im_pil = Image.fromarray(zoomed_face)
                zoomed_face_tensor = convert_tensor(transforms.Grayscale(num_output_channels=1)(im_pil))
                zoomed_face_tensor.unsqueeze_(0)

                # Get the prediction from the model
                emotion_prediction_tensor = model(zoomed_face_tensor)
                confidence, predicted = torch.max(emotion_prediction_tensor, 1)
                emotion_predicted = LabelMap.getLabel(predicted.item())

                # Label the bounding box with the prediction
                cv2.putText(
                    img = frame, 
                    text = f"{emotion_predicted} {confidence[0].item() * 100:.1f}%", 
                    org = (x, y - 10), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = .5, 
                    color = (0, 255, 0), 
                    thickness = 2
                    )
            return frame

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on("image")
def receive_image(image):
    # Decode the base64-encoded image data
    image = base64_to_image(image)

    # Perform image processing using OpenCV
    # frame_resized = cv2.resize(image, (640, 360))
    frame_resized = gen_frames(image)

    # Encode the processed image as a JPEG-encoded base64 string
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()

    # Prepend the base64-encoded string with the data URL prefix
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data

    # Send the processed image back to the client
    emit("processed_image", processed_img_data)


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    socketio.run(app, debug=False)
    # app.run(debug=False)