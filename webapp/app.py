from flask import Flask, request, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = tf.keras.models.load_model("cifar10_model.h5")
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    # Save a larger copy for display
    display_img = Image.open(file).resize((256, 256), Image.NEAREST)
    filepath = os.path.join("static", file.filename)
    display_img.save(filepath)

    # Preprocess a 32x32 version for the model
    img_array = np.array(display_img.resize((32,32))).astype("float32")/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = round(100 * np.max(prediction), 2)

    # Render template with results and larger photo
    return render_template("index.html",
                           prediction=classes[class_idx],
                           confidence=confidence,
                           uploaded_image=url_for("static", filename=file.filename))


if __name__ == "__main__":
    app.run(debug=True)
