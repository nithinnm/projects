from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model
model_path = 'nm_model.h5'
model = tf.keras.models.load_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        img = Image.open(file)
        img = img.resize((300, 300))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = int(round(model.predict(img_array)[0][0]))

        # Determine the result based on the prediction
        result = "Syndrome" if prediction != 1 else "Normal"

        return jsonify ({"result": result})
    except Exception as e:
        return jsonify ({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
