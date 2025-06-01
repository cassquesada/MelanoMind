
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Cargamos el modelo entrenado
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_model()

# Diccionario de clases (número -> nombre)
classes = {
    0: ('akiec', 'Actinic keratoses y carcinoma in situ'),
    1: ('bcc', 'Carcinoma de células basales'),
    2: ('bkl', 'Lesiones benignas tipo queratosis'),
    3: ('df', 'Dermatofibroma'),
    4: ('nv', 'Nevus melanocítico (lunar)'),
    5: ('vasc', 'Lesiones vasculares'),
    6: ('mel', 'Melanoma (⚠️ cáncer de piel)')
}

st.set_page_config(page_title="Detector de Cáncer de Piel", layout="centered")
st.title("🔬 Detector de Cáncer de Piel")
st.write("Subí una imagen de una mancha o lesión en la piel y te diremos qué podría ser.")

uploaded_file = st.file_uploader("Elegí una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Imagen subida', use_column_width=True)

    # Procesar la imagen
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer la predicción
    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label, predicted_description = classes[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.markdown("---")
    st.subheader("🧠 Resultado de la predicción")
    st.success(f"Predicción: {predicted_label.upper()} — {predicted_description}")
    st.info(f"Confianza del modelo: {confidence:.2f}%")

    st.markdown("---")
    st.write("Este modelo es educativo. Si tenés dudas reales, consultá a un dermatólogo.")
