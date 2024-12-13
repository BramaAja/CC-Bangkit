from flask import Flask, request, jsonify
from ultralytics import YOLO
import os
from flask_cors import CORS  # Impor flask_cors

app = Flask(__name__)

# Menambahkan CORS pada aplikasi Flask
CORS(app)  # Menambahkan CORS untuk semua rute dan asal (origin)

# Load YOLOv8 model
model = YOLO('best.pt')  # Ganti path dengan lokasi best.pt Anda

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)

    # Perform inference
    results = model.predict(source=image_path, conf=0.4)

    # Extract bounding box predictions
    predictions = []
    for result in results:
        if result.boxes.xyxy.shape[0] > 0:  # Periksa apakah ada prediksi
            for i, box in enumerate(result.boxes.xyxy.numpy()):
                confidence = result.boxes.conf[i].item()  # Ambil confidence
                class_id = int(result.boxes.cls[i].item())  # Ambil class id
                
                predictions.append({
                    'x1': float(box[0]), 'y1': float(box[1]), 'x2': float(box[2]), 'y2': float(box[3]),
                    'confidence': float(confidence), 'class': class_id
                })
        else:
            predictions.append({
                'error': 'No objects detected'
            })

    os.remove(image_path)  # Clean up uploaded file

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    
    app.run(host='0.0.0.0', port=5000, debug=True)
