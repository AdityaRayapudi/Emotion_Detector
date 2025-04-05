from flask import Flask, render_template, Response
import cv2
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from EmotionDetection import NeuralNetwork, LabelMap

#Initialize the Flask app
app = Flask(__name__)

# Initalize pytorch model
model = NeuralNetwork()
model.load_state_dict(torch.load("base_emotion_model.pth", weights_only=True))

# Initalize cv2 face detection
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Get camera  capture
camera = cv2.VideoCapture(0)
convert_tensor = transforms.ToTensor()

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        cv2.flip(frame, 1)
        if not success:
            break
        else:
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
                    fontScale = 1.25, 
                    color = (0, 255, 0), 
                    thickness = 2
                    )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(debug=False)