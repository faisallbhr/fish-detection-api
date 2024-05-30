from flask import Flask, request, send_file, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid

app = Flask(__name__)
model = YOLO('./model.pt')

SAVE_DIR = 'tmp'
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image uploaded'}), 400
    
    try:
        image = Image.open(io.BytesIO(file.read()))
        results = model(image)

        unique_filename = str(uuid.uuid4()) + '.jpg'
        result_image_path = os.path.join(SAVE_DIR, unique_filename)

        result = results[0]
        result.save(result_image_path)

        # return img file
        # return send_file(result_image_path, mimetype='image/jpeg')
        
        # return img path
        return jsonify({'image_path': unique_filename})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/results/<filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_from_directory(SAVE_DIR, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
