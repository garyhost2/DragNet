from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI()


# Load the ONNX model
try:
    session = ort.InferenceSession("./DragNet.onnx")
except Exception as e:
    print("Error loading model:", e)
    raise

def preprocess_image(image: Image.Image):
    # Resize image to the input size expected by your model (e.g., 224x224)
    image = image.resize((640, 640))
    image_array = np.array(image).astype(np.float32)
    # Convert grayscale to RGB if needed
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    # Change data layout from HWC to CHW
    image_array = image_array.transpose(2, 0, 1)
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate that the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing image file")

    # Preprocess the image to match the model's input requirements
    input_data = preprocess_image(image)
    # Prepare the input dictionary for ONNX Runtime
    inputs = {session.get_inputs()[0].name: input_data}
    
    # Run inference
    outputs = session.run(None, inputs)
    
    # Process and format the output (assuming outputs[0] is your prediction)
    result = outputs[0].tolist()
    return {"prediction": result}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
