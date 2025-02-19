import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Resize as per model's input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array
def predict_image(model, img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

model = tf.keras.models.load_model("cats_dogs_classifier.h5")

Tk().withdraw()  # Hide the root window
img_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

if img_path:
    result = predict_image(model, img_path)
    print(f"The uploaded image is a: {result}")
else:
    print("No image selected.")
