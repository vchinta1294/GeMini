# download_models.py
import os
import torch
from transformers import BertModel, ViTModel
from huggingface_hub import snapshot_download

def download_models():
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Download text model
    print("Downloading PubMedBERT...")
    text_model_path = "models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    if not os.path.exists(text_model_path):
        # Use huggingface_hub to download
        snapshot_download(repo_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", 
                         local_dir=text_model_path)
    
    # Download ViT models
    print("Downloading ViT-16...")
    img_model_path16 = "models/vit-base-patch16-224-in21k"
    if not os.path.exists(img_model_path16):
        snapshot_download(repo_id="google/vit-base-patch16-224-in21k", 
                         local_dir=img_model_path16)
    
    print("Downloading ViT-32...")
    img_model_path32 = "models/vit-base-patch32-224-in21k"
    if not os.path.exists(img_model_path32):
        snapshot_download(repo_id="google/vit-base-patch32-224-in21k", 
                         local_dir=img_model_path32)
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    download_models()
