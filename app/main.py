from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from starlette.requests import Request
from PIL import Image
import io

import numpy as np
from tensorflow import keras
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('/model/Mymodel.keras')
class_names =  ['Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 'Banana Black Sigatoka Disease', 'Banana Bract Mosaic Virus Disease', 'Banana Healthy Leaf', 'Banana Insect Pest Disease', 'Banana Moko Disease', 'Banana Panama Disease', 'Banana Yellow Sigatoka Disease', 'Blueberry healthy', 'Cherry Powdery mildew', 'Cherry healthy', 'Corn  Cercospora leaf spot Gray leaf spot', 'Corn  Common rust ', 'Corn  Northern Leaf Blight', 'Corn  healthy', 'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Orange Haunglongbing', 'Peach Bacterial spot', 'Peach healthy', 'Pepper, bell Bacterial spot', 'Pepper, bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Rice Brown Spot', 'Rice Healthy', 'Rice Hispa', 'Rice Leaf Blast', 'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus', 'Tomato healthy', 'paddy bacterial leaf blight', 'paddy bacterial leaf streak', 'paddy bacterial panicle blight', 'paddy blast', 'paddy brown spot', 'paddy dead heart', 'paddy downy mildew', 'paddy healthy', 'paddy hispa', 'paddy tungro']

@app.get("/{name}")
def hello(name):
    return{"Hello {} and welcome to this API".format(name)}

@app.get("/")
def greet():
    return{"Hello"}

if __name__=="__main__":
    uvicorn.run(app)
    
@app.post("/predict")
async def predict(request: Request, pic: UploadFile = File(...)):
    try:
        # Read image file
        contents = await pic.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image (adjust as needed based on your CNN model)
        image = image.resize((224, 224))  # Adjust the size as per your model requirements
        image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]

        # Expand dimensions to match the input shape expected by the model
        input_image = np.expand_dims(image_array, axis=0)

        # Perform inference
        predictions = model.predict(input_image)

        # Get the predicted class
        predicted_class = np.argmax(predictions)
        class_name = class_names[predicted_class]

        return JSONResponse(content={"prediction": class_name})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)