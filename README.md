# Signature_Recognition: Unlocking Identity through Signature Verification with Vision Transformers

Welcome to the **Signature_Recognition** project! This innovative endeavor leverages the power of the Vision Transformer (ViT) model to accurately recognize signatures from a diverse set of 16 users. By utilizing cutting-edge machine learning techniques, we aim to streamline the signature verification process, making it more efficient and reliable.

## 🌟 Project Overview

In a world where digital signatures play a crucial role in authentication, our project harnesses the capabilities of the ViT model, which is pre-trained on the vast ImageNet-21k dataset. The process of recognizing signatures involves a few essential steps:
1. **Preprocessing**: Each signature image is resized, normalized, and transformed into a tensor to prepare it for analysis.
2. **Inference**: The preprocessed image is fed into our ViT model, which predicts the user’s identity based on the unique characteristics of their signature.
3. **Mapping Predictions**: The model's output class index is then linked to the respective user name, providing a seamless user experience.

## 🚀 Requirements

Before diving into the project, ensure you have the following installed on your local machine:

- **Python 3.x**
- `torch`
- `torchvision`
- `transformers`
- `Pillow` (PIL)

### 🛠 Installation

Get started by installing the necessary dependencies with:

```bash
pip install torch torchvision transformers pillow
```

### 🎉 Pretrained Model

We utilize the robust `ViTForImageClassification` architecture, fine-tuned specifically for signature recognition. Don’t forget to download your fine-tuned weights and place them in the designated directory!

## 🏁 How to Run

### 1. Clone the Repository

Begin your journey by cloning this repository:

```bash
git clone https://github.com/Naman-Sahni/Signature_Recognition.git
cd Signature_Recognition
```

### 2. Place the Model Weights

Ensure your model weights are saved as `modeler.pth` in the `/content/` directory. Adjust the path in the code if you’ve placed it elsewhere.

### 3. Image Preparation

Your input image should be in BMP format. Position your image in the `/content/` directory or update the `image_path` in the script to point to your image.

### 4. Running the Code

With everything in place, run the Python script for inference:

```bash
python classify_signature.py
```

The script will automatically load the signature image, preprocess it, and run it through the ViT model to reveal the identity behind the signature!

## 🔍 Code Breakdown

- **Model Definition**: We load the powerful Vision Transformer (ViT) model from Hugging Face's library, configured for 16 distinct user signatures.
- **Image Preprocessing**: Each image is resized to (224, 224), normalized with ImageNet standard values, and converted into a tensor for optimal processing.
- **Model Inference**: The model processes the preprocessed image and generates logits, determining the predicted class based on the highest value.
- **Mapping Class to User Name**: The predicted class index is mapped to the respective user name, making the output easy to understand and actionable.

## ✨ Customization

Want to make it your own? Feel free to:
- **Add or Modify User Categories**: Update the `user_names` list and adjust the `num_labels` parameter accordingly.
- **Experiment with Datasets**: If you're using a different dataset or image format, tweak the `preprocess_image()` function to fit your needs!


