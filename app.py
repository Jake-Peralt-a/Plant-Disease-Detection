import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
from transformers import ViTForImageClassification, ViTConfig

def load_model(model_path, num_labels=2):

    config = ViTConfig.from_pretrained('google/vit-base-patch16-224', num_labels=num_labels)
    model = ViTForImageClassification(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    return model



MODEL_PATH = 'C:\MyFarm\plant_disease_model.pth'  # Update this path to the .pth file path on your PC
model = load_model(MODEL_PATH)
class_names = ['Bacterial Spot', 'Healthy']
st.title("Plant Disease Prediction")
st.write("Drag and drop an image, and the model will predict the class.")

uploaded_file = st.file_uploader("", type=["jpg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Predicting...")

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  


    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        predicted_class = class_names[predicted.item()]
    
    st.write(f'Predicted Class: {predicted_class}')
    st.write(f'Confidence: {confidence.item() * 100:.2f}%')
