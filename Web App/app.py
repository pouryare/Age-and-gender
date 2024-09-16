from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from model import image_pre, predict

app = Flask(__name__)
UPLOAD_FOLDER = '/static'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file1' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file1 = request.files['file1']
    if file1.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file1 and allowed_file(file1.filename):
        filename = 'input.' + file1.filename.rsplit('.', 1)[1].lower()
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file1.save(path)
        data = image_pre(path)
        age, gen = predict(data)
        gender = 'Male' if gen == 1 else 'Female'
        result = f'Predicted age is {age} years and the person is a {gender}'
        return jsonify({'result': result}), 200
    else:
        return jsonify({'error': 'File type not allowed'}), 400

if __name__ == "__main__":
    app.run(debug=True)
