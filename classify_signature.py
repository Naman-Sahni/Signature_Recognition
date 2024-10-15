import tensorflow
import keras
import torchvision
import torch
from transformers import ViTForImageClassification
from torchvision import transforms
from PIL import Image

# Define data transformations
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to (224, 224)
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Example preprocess_image function
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB mode
    image = data_transform(image)  # Apply the defined transformations
    return image.unsqueeze(0)  # Add batch dimension

# Initialize the model architecture (assuming the same configuration as during training)
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                  num_labels=16)  # Assuming 16 users

# Specify the path to the .pth file containing the model weights
model_path = r"path"  # Adjust based on your extracted path

# Load the model weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Adjust map_location as needed

# Evaluate mode (important if your model has layers like Dropout or BatchNorm that behave differently during training and evaluation)
model.eval()

# Example usage for inference
image_path = r"path"  # Replace with the actual path to your image
input_image = preprocess_image(image_path)

# Perform inference
with torch.no_grad():
    outputs = model(input_image)

# Get predicted class index
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()

# Example: Assume you have a list of user names corresponding to class indices
user_names = ['Aditya', 'Ayushi', 'deepak', 'deepam','devesh', 'harsh', 'himarshini ', 'Moksh', 'Nikita', 'ninja', 'nishant', 'priydarshini', 'rajsabi', 'rishab', 'rudraksh', 'shomesh']

# Get predicted user name
predicted_user_name = user_names[predicted_class_idx]

print(f"Predicted user name: {predicted_user_name}")
