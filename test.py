import os
import torch
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor

# Load environment variables from .env file
load_dotenv()

# Get the path to the dataset folder from the environment variable
TEST_FOLDER = os.getenv('CEC_2025_dataset')

# Check if the environment variable is set
if TEST_FOLDER is None:
    raise ValueError("Environment variable 'CEC_2025_dataset' is not set. Please set it to the test folder path.")

# Print dataset location for debugging
print(f"Dataset folder is located at: {TEST_FOLDER}")

# Load the model path from the environment variable
model_path = os.getenv("MODEL")

# Check if the environment variable is set
if not model_path:
    raise ValueError("Environment variable 'MODEL' is not set or is empty.")

# Define the output CSV file
OUTPUT_FILE = "results.csv"

# Load the trained ViT model (CPU only)
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Load the feature extractor for ViT
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Define the image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to ViT model input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

# Function to preprocess and predict an image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert image to RGB
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return "yes" if predicted_class == 1 else "no"

# Get all test images (Ensure sorting is correct)
test_images = sorted(
    [f for f in os.listdir(TEST_FOLDER) if f.startswith("test__") and f.endswith(('.png', '.jpg'))],
    key=lambda x: int(x.split('__')[-1].split('.')[0])  # Sort numerically
)

# Check if output CSV exists to avoid duplicate processing
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    processed_images = set(existing_df["Test Image"].tolist())
else:
    existing_df = pd.DataFrame(columns=["Test Image", "Result"])
    processed_images = set()

# Run inference and collect results
results = []
for filename in test_images:
    if filename in processed_images:
        print(f"Skipping already processed file: {filename}")
        continue  # Skip images that are already in the CSV

    image_path = os.path.join(TEST_FOLDER, filename)
    prediction = predict_image(image_path)
    results.append([filename, prediction])

    # Append to CSV immediately (saves progress)
    df = pd.DataFrame([[filename, prediction]], columns=["Test Image", "Result"])
    df.to_csv(OUTPUT_FILE, mode="a", header=not os.path.exists(OUTPUT_FILE), index=False)

    print(f"Processed and saved: {filename}")

print(f"\nAll results saved to {OUTPUT_FILE}")
