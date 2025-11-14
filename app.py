from flask import Flask, request, render_template
from cnnClassifier.pipeline.prediction_pipeline import PredictionPipeline
import os

app = Flask(__name__)
pipeline = PredictionPipeline()  # load model once


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded!"

    file = request.files["file"]

    if file.filename == "":
        return "Empty file!"

    # Save uploaded image temporarily
    upload_path = os.path.join("static", "uploaded.jpg")
    file.save(upload_path)

    # Run prediction
    result = pipeline.predict(image_path=upload_path)

    # Pass results to UI
    return render_template("index.html", image_path=upload_path, result=result)


if __name__ == "__main__":
    app.run(debug=True)
