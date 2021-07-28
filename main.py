import flask 
from flask import render_template,Response
import cv2
import time
from flask.app import Flask
from video_feed import video_feeding

api = Flask(__name__)

@api.route('/')
def index():
    return render_template('index.html')

@api.route('/detect')
def detect():
    return Response(video_feeding(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    api.run(port=5000,host='0.0.0.0',threaded=True,debug=True)