import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Function to load and preprocess the image for prediction
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict pneumonia
def predict_pneumonia(img_path, model):
    img_array = load_and_preprocess_image(img_path)
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        result = "PNEUMONIA"
        confidence = prediction[0][0] * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction[0][0]) *100
    return result, confidence

# Function to provide precautions and medication advice for pneumonia
def pneumonia_precautions_and_medication():
    precautions = """
    **Precautions for Pneumonia:**
    - Rest: Make sure to get plenty of rest to help your body fight off the infection.
    - Hydration: Drink plenty of fluids (water, soup, etc.) to stay hydrated.
    - Healthy Nutrition: Eat a balanced diet to support your immune system.
    - Avoid Smoking: If you smoke, consider quitting, as it can worsen pneumonia.
    - Stay Isolated: Avoid close contact with others to prevent spreading the infection.
    - Follow your doctor's advice regarding antibiotics or other treatments.
    """
    
    medications = """
    **Proposed Medications (Consult your healthcare provider):**
    - Antibiotics (e.g., Amoxicillin, Azithromycin, or others based on bacterial cause).
    - Antiviral medications if pneumonia is caused by a virus (e.g., Oseltamivir for influenza).
    - Over-the-counter pain relievers for fever and discomfort (e.g., Paracetamol, Ibuprofen).
    - Cough medications (e.g., Dextromethorphan or Guaifenesin) for managing cough.
    - Oxygen therapy may be required in severe cases to improve breathing.
    """
    
    return precautions, medications

# Streamlit UI elements
st.title('Pneumonia Detection from Chest X-rays')
st.write("This application uses a trained deep learning model to detect pneumonia in chest X-ray images.")

# Upload image file
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    img_path = os.path.join("uploaded_images", uploaded_file.name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    st.image(img_path, caption="Uploaded Image", use_column_width=True)
    
    # Load the model (make sure it's in the same directory or specify path)
    model = load_model('my_model.keras')  # Adjust path as needed
    
    # Make prediction
    result, confidence = predict_pneumonia(img_path, model)
    
    # Show result and confidence
    st.write(f"*Prediction:* {result}")
    st.write(f"*Confidence:* {confidence:.2f}%")
    
    # Provide an explanation about the result
    if result == "PNEUMONIA":
        st.write("The model detected signs of pneumonia. Please consult a healthcare professional for further evaluation.")
        
        # Display precautions and medication advice
        precautions, medications = pneumonia_precautions_and_medication()
        st.write(precautions)
        st.write(medications)
    else:
        st.write("The model did not detect pneumonia. However, it is always advisable to consult a healthcare professional for a more accurate diagnosis.")

# Add a footer or additional information
st.sidebar.title("About")
st.sidebar.info("Pneumonia is an infection that inflames the air sacs in one or both lungs, which can fill with fluid or pus, causing symptoms such as coughing, fever, chills, and difficulty breathing. It can be caused by a variety of organisms, including bacteria, viruses, fungi, and parasites. Common causes of bacterial pneumonia include Streptococcus pneumoniae and Haemophilus influenzae, while viral pneumonia is often caused by influenza viruses, coronaviruses, or respiratory syncytial virus (RSV).")
