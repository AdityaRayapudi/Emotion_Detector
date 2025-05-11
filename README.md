# Emotion Detector

A flask application that uses a webcam and a PyTorch model to predict the users emotions


**Windows Git Bash**:

```bash
python -m venv env
./env/scripts/activate
pip install -r requirements.txt
gunicorn --worker-class eventlet app:app
```

**Mac OS / Linux**:

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
gunicorn --worker-class eventlet app:app
```


The app will be running on (http://127.0.0.1:8000)