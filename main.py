import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torchvision.models import mobilenet_v2

class CustomMobileNetv2(nn.Module):
    def _init_(self, output_size):
        super()._init_()
        self.mnet = mobilenet_v2(pretrained=True)
        self.freeze()

        self.mnet.classifier = nn.Sequential(
            nn.Linear(1280, output_size),
            nn.LogSoftmax()
        )

    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = True

# Function to preprocess the uploaded image
def process_uploaded_image(image):
    image = image.convert("RGB")
    image = test_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Streamlit app
st.title("Reef Watch App")

# Load the pre-trained model
model_path = "uas_DL.pth"  # Replace with the path where you saved your model
model = torch.load(model_path, map_location=torch.device('cpu')).to(torch.float32)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Button to make predictions
    if st.button("Predict"):
        # Make predictions
        processed_image = process_uploaded_image(image)
        processed_image = processed_image.to(device, dtype=torch.float32)
        with torch.no_grad():
            model.eval()
            output = model(processed_image)
            _, predicted_class = torch.max(output, 1)
            # Replace with labels according to your model
            label2cat = ['bleaching', 'non-bleaching']
            predicted_label = label2cat[predicted_class.item()]
            st.write("Prediction:", predicted_label)