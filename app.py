from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import torch
from torchvision import transforms
from PIL import Image
import json
from transformers import ViTForImageClassification, ViTConfig

app = Flask(__name__)

# Project Title
PROJECT_TITLE = "Brainiac: Tumor Detection AI"

# Folder to store uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model configuration and weights
def load_model(model_path=os.getenv("MODEL")):
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
    config.num_labels = 2  # Assuming binary classification

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        config=config,
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the model globally to avoid reloading -ACTS LIKE A CACHE
model = load_model()

# Function to preprocess the image
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    return input_tensor

# Function to predict tumor
def predict_tumor(image_path):
    # Preprocess the image
    input_tensor = preprocess_image(image_path)

    # Make a prediction
    with torch.no_grad():
        outputs = model(input_tensor).logits
        probs = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(probs, dim=1)

    # Map the prediction to a label
    labels = ["no tumor", "tumor"]
    prediction_label = labels[preds.item()]
    confidence_value = confidence.item()

    # Create the JSON output
    result = {
        "pred": prediction_label,
        "confidence": confidence_value,
        "image": image_path
    }

    # Convert the result to JSON format
    result_json = json.dumps(result, indent=4)

    return result_json

# Load the page
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"

        file = request.files["file"]
        if file.filename == "":
            return "No selected file"

        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Process the image and predict
            prediction = predict_tumor(file_path)
            return render_template("index.html", title=PROJECT_TITLE, filename=filename, prediction=json.loads(prediction))

    return render_template("index.html", title=PROJECT_TITLE, filename=None, prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
