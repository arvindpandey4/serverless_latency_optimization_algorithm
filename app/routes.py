import os
from flask import render_template, request, redirect, url_for, send_from_directory
from app import app
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np


model_path = os.path.join('model', 'cat_dog_classifier.h5')


def load_trained_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        return model
    except ValueError as e:
        print(f"Error loading model: {e}")
        raise

model = load_trained_model(model_path)

def predict_image(image_path):
    image = load_img(image_path, target_size=(64, 64))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  # Normalize the image
    prediction = model.predict(image)
    probability = prediction[0][0]
    label = 'dog' if probability > 0.5 else 'cat'
    accuracy = probability if label == 'dog' else 1 - probability
    return label, accuracy

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            uploads_dir = os.path.join(app.root_path, 'uploads')
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            filepath = os.path.join(uploads_dir, filename)
            file.save(filepath)
            label, accuracy = predict_image(filepath)  # <--- Updated to get label and accuracy
            return render_template('result.html', label=label, accuracy=accuracy, filename=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join(app.root_path, 'uploads'), filename)
