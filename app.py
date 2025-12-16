from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from keras.models import load_model           
from keras.utils import load_img, img_to_array  
from remedies import REMEDIES

# Initialize Flask app
app = Flask(__name__)

# Use relative path 
MODEL_PATH = os.path.join(os.path.dirname(__file__), "skin_disease_model.h5")
model = load_model(MODEL_PATH)

# Classes in same order as training dataset
CLASSES = [
    "Acne",
    "Actinic Keratosis",
    "Benign Tumors",
    "Bullous",
    "Candidiasis",
    "Drug Eruption",
    "Eczema",
    "Infestations/Bites",
    "Lichen",
    "Lupus",
    "Moles",
    "Psoriasis",
    "Rosacea",
    "Seborrheic Keratoses",
    "Skin Cancer",
    "Sun/Sunlight Damage",
    "Tinea",
    "Unknown/Normal",
    "Vascular Tumors",
    "Vasculitis",
    "Vitiligo",
    "Warts"
]

# Folder for uploaded images
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/work")
def work():
    return render_template("work.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Get uploaded image
        file = request.files.get("image")
        if not file:
            return jsonify({"success": False, "error": "No image uploaded"})

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        #  Preprocess image 
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions) * 100)
        predicted_class = CLASSES[np.argmax(predictions)]

        # Remedies 
        remedies = REMEDIES.get(predicted_class, ["No remedies available."])

        # Extra form data
        symptoms = request.form.getlist("symptoms")
        duration = request.form.get("duration")

        result = {
            "success": True,
            "condition": predicted_class,
            "confidence": round(confidence, 2),
            "recommendations": remedies,
            "symptoms": symptoms,
            "duration": duration
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
