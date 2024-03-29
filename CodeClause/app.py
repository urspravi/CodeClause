from flask import Flask, render_template, request
from werkzeug.utils import secure_filename  # Import secure_filename function
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the pre-trained deep learning model
model = load_model('model.h5')

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250))  # Resize image to match model input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # Use secure_filename here
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the uploaded image
        processed_image = preprocess_image(file_path)
        
        # Make prediction using the model
        prediction = model.predict(processed_image)
        
        # Assuming prediction is an array of probabilities
        max_probability = prediction.max()

        # Assuming your classes are ordered such that "Diseased" corresponds to class 1
        predicted_class = "Diseased" if max_probability > 0.9 else "Healthy"

        
        return render_template('index.html', message='Prediction: ' + predicted_class , filename=filename)
    
    else:
        return render_template('index.html', message='Invalid file format')

if __name__ == '__main__':
    app.run(debug=True)
