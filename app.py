from flask import Flask , send_file , request
from utility import *  
from run_pipeline import online_run
import uuid
import os
from flask import Flask, send_file, request, render_template


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

@app.route('/health')
def health():
    return {'status': 'ok'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate',methods = ['POST'])
def generate():
    recording = request.files['recording']
    input_path = os.path.join(UPLOAD_FOLDER, f'{uuid.uuid4()}.wav')
    recording.save(input_path)
    return send_file(online_run(input_path)) ## online run returns the path of the generated filr

if __name__ == '__main__':
    app.run(debug=True)