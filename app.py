from flask import Flask, request, send_file, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import io
import os
import uuid

app = Flask(__name__)
model = YOLO('./model.pt')

IMAGE_DIR = 'images'
TEXT_DIR = 'texts'
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

def parse_result_to_json(result_txt_path):
    with open(result_txt_path, 'r') as file:
        lines = file.readlines()
    
    results = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            probability = parts[0]
            label = ' '.join(parts[1:])
            results.append({'label': label, 'probability': float(probability)})
    
    return results

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

        unique_image_filename = str(uuid.uuid4()) + '.jpg'
        result_image_path = os.path.join(IMAGE_DIR, unique_image_filename)
        result = results[0]
        result.save(result_image_path)

        unique_text_filename = str(uuid.uuid4()) + '.txt'
        result_txt_path = os.path.join(TEXT_DIR, unique_text_filename)
        result.save_txt(result_txt_path)

        prediction_results = parse_result_to_json(result_txt_path)
        
        return jsonify({'image_path': unique_image_filename, 'predictions': prediction_results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/results/<filename>', methods=['GET'])
def get_image(filename):
    try:
        return send_from_directory(IMAGE_DIR, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
