import tensorflow as tf
from utils.preprocess import preprocess_image

# Clear previous sessions
tf.keras.backend.clear_session()

# ✅ Load fixed model (.h5)
model = tf.keras.models.load_model(
    "model/model.h5",
    compile=False
)

def predict(image):
    img = preprocess_image(image)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        label = "Fake"
        confidence = pred
    else:
        label = "Real"
        confidence = 1 - pred

    return label, float(confidence)