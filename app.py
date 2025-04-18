import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask import Flask, render_template, request, redirect

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Define the target size used during training
target_size = (224, 224)

# Class labels
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

app = Flask(__name__)

def load_and_preprocess_image(img_path):
    """Load and preprocess an image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class_name = None
    confidence = None
    img_filename = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded file to a temporary location
        img_filename = file.filename
        img_path = os.path.join('static', img_filename)
        file.save(img_path)

        # Load and preprocess the image
        preprocessed_image = load_and_preprocess_image(img_path)

        # Make prediction
        prediction = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_labels[predicted_class_index]
        confidence = round(np.max(prediction) * 100, 2)  # Convert to percentage

    return render_template('index.html', tumor_type=predicted_class_name, confidence=confidence, filepath=img_filename)

if __name__ == "__main__":
    app.run(debug=True)
