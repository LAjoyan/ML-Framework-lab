from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="CIFAR-10 Classifier (ONNX)")

# Load the model once when the server starts
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

@app.get("/")
def health_check():
    return {"status": "Online", "framework": "ONNX", "best_accuracy": "73.85%"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Preprocess: Resize to 32x32 and normalize
    img = img.resize((32, 32))
    img_array = np.array(img).astype(np.float32) / 255.0

    # Change shape from (32, 32, 3) to (1, 3, 32, 32)
    img_array = np.transpose(img_array, (2, 0, 1))
   

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(3, 1, 1)
    img_array = (img_array - mean) / std

    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Run Inference
    outputs = session.run(None, {input_name: img_array})

    logits = outputs[0][0]

    exp_logits = np.exp(logits - np.max(logits)) # Subtract max for numerical stability
    probs = exp_logits / exp_logits.sum()

    prediction = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return {
        "class_id": prediction,
        "label": CLASSES[prediction],
        "confidence": f"{confidence:.2%}"
    }
