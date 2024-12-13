import pickle
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import numpy as np
import re
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download('stopwords')

# Stopwords list
stop_words = set(stopwords.words('english'))

# Remove noise
def remove_noise(text):
    text = re.sub(r'<.*?>', '', str(text))  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)    # Remove punctuation
    return text

# Remove stopwords
def remove_stopwords(text):
    words = text.split()  # Split text into words
    filtered_words = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
    return ' '.join(filtered_words)  # Join filtered words back into a string

app = Flask(__name__)

# Menambahkan CORS ke aplikasi Flask
CORS(app)  # Ini memungkinkan semua domain untuk mengakses API

# Memuat model .h5
model = tf.keras.models.load_model('best3.h5')

# Memuat tokenizer yang telah dilatih
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Membaca daftar label class (focus_area) dari file CSV
focus_area_file = 'cat_penyakit.csv'
focus_area_df = pd.read_csv(focus_area_file)
label_list = focus_area_df['focus_area'].tolist()  # Kolom 'Focus_Area' mengandung daftar label

@app.route('/predict', methods=['POST'])
def predict():
    # Periksa apakah input teks ada di body permintaan
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400

    input_text = request.json['text']  # Ambil teks dari request

    try:
        # Bersihkan teks dengan menghapus noise dan stopwords
        clean_text = remove_noise(input_text)
        clean_text = remove_stopwords(clean_text)

        # Preprocess input text (gunakan tokenizer yang sudah dilatih)
        max_length = 100  # Panjang maksimal input teks
        sequences = tokenizer.texts_to_sequences([clean_text])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=max_length, padding='post', truncating='post'
        )

        # Perform inference
        predictions = model.predict(padded_sequences)
        predicted_class = np.argmax(predictions[0])  # Ambil kelas dengan probabilitas tertinggi
        confidence = float(predictions[0][predicted_class])  # Probabilitas untuk kelas tersebut

        # Convert predicted class index to label
        predicted_label = label_list[predicted_class]

        response = {
            'input_text': input_text,
            'clean_text': clean_text,
            'predicted_class': int(predicted_class),
            'predicted_label': predicted_label,
            'confidence': confidence
        }

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
