from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("cifar10_model.h5")

# CIFAR-10 class labels
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).resize((32,32))  # CIFAR-10 images are 32x32
    img_array = np.array(img).astype("float32")/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return jsonify({"class": classes[class_idx], "confidence": confidence})

if __name__ == "__main__":
    app.run(debug=True)
