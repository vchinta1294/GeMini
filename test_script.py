# test_script.py
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for servers without display
import matplotlib.pyplot as plt
from transformers import BertTokenizer, ViTImageProcessor, PretrainedConfig
from src.models import GEMINI_scratch
import random

def load_pretrained_model():
    """Load the pre-trained GeMini model"""
    checkpoint_path = "output/GEMINI/pretrained"
    
    # Check if checkpoint exists
    if not os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")):
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        print("Please make sure you've downloaded the pre-trained model from the provided Google Drive link.")
        return None
    
    try:
        # Load the model from pretrained checkpoint
        print(f"Loading model from {checkpoint_path}...")
        model = GEMINI_scratch.from_pretrained(checkpoint_path)
        model.eval()
        print("✓ Successfully loaded model")
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        # Try a different approach if the first one fails
        try:
            print("Attempting alternative loading method...")
            # Load config first
            config = PretrainedConfig.from_pretrained(checkpoint_path)
            # Initialize model with config
            model = GEMINI_scratch(config)
            # Load state dict manually
            state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            print("✓ Successfully loaded model with alternative method")
            return model
        except Exception as e2:
            print(f"Error with alternative method: {e2}")
            return None

def load_sample_images(mimic_cxr_dir, num_samples=5):
    """Load sample images from the MIMIC-CXR dataset"""
    # Path to the files directory
    files_dir = os.path.join(mimic_cxr_dir, 'files')
    
    # Get a list of all p* directories (patient directories)
    p_dirs = [d for d in os.listdir(files_dir) if d.startswith('p') and os.path.isdir(os.path.join(files_dir, d))]
    
    sample_images = []
    
    # Randomly select some patient directories
    for p_dir in random.sample(p_dirs, min(10, len(p_dirs))):
        p_path = os.path.join(files_dir, p_dir)
        
        # Get patient directories
        patient_dirs = [d for d in os.listdir(p_path) if os.path.isdir(os.path.join(p_path, d))]
        
        for patient_dir in patient_dirs:
            patient_path = os.path.join(p_path, patient_dir)
            
            # Get study directories
            study_dirs = [d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]
            
            for study_dir in study_dirs:
                study_path = os.path.join(patient_path, study_dir)
                
                # Get image files
                image_files = [f for f in os.listdir(study_path) if f.endswith('.jpg')]
                
                for image_file in image_files:
                    image_path = os.path.join(study_path, image_file)
                    sample_images.append(image_path)
                    
                    if len(sample_images) >= num_samples:
                        return sample_images
    
    return sample_images

def test_gemini_on_samples(model, image_paths):
    """Test the GeMini model on sample images"""
    if model is None:
        print("No model loaded. Cannot test.")
        return
    
    # Load tokenizers and processors
    tokenizer = BertTokenizer.from_pretrained("models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    vit_processor16 = ViTImageProcessor.from_pretrained("models/vit-base-patch16-224-in21k")
    vit_processor32 = ViTImageProcessor.from_pretrained("models/vit-base-patch32-224-in21k")
    
    # Process each image
    for i, image_path in enumerate(image_paths):
        print(f"\nProcessing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process with ViT processors
            inputs16 = vit_processor16(images=image, return_tensors="pt")
            inputs32 = vit_processor32(images=image, return_tensors="pt")
            
            # Create dummy inputs for other modalities
            dummy_text = "This is a chest X-ray."
            text_inputs = tokenizer(dummy_text, return_tensors="pt")
            
            # Forward pass with minimal required inputs
            with torch.no_grad():
                outputs = model(
                    input_ids=text_inputs['input_ids'],
                    attention_mask=text_inputs['attention_mask'],
                    pixel_values16=inputs16['pixel_values'],
                    pixel_values32=inputs32['pixel_values'],
                )
            
            # Save image to file
            save_path = f"sample_image_{i+1}.png"
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title(f"Sample Image {i+1}")
            plt.axis('off')
            plt.savefig(save_path)
            plt.close()
            print(f"Image saved to {save_path}")
            
            # Display features
            if hasattr(outputs, 'pooler_output'):
                print(f"Feature vector shape: {outputs.pooler_output.shape}")
            elif hasattr(outputs, 'hidden_states'):
                print(f"Hidden states shape: {outputs.hidden_states.shape}")
            
            # If there are logits, display them
            if hasattr(outputs, 'logits') and outputs.logits is not None:
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
                print(f"Predicted probabilities for top conditions:")
                for j in range(min(5, len(probabilities[0]))):
                    print(f"  Class {j}: {probabilities[0][j]:.4f}")
        
        except Exception as e:
            print(f"Error processing image: {e}")
    
    print("\nModel inference completed on sample images")

def main():
    # Set the path to your MIMIC-CXR dataset
    mimic_cxr_dir = '/home/vchinta/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0'
    
    print("=== Testing GeMini on Public Dataset ===")
    
    # Load the pre-trained model
    model = load_pretrained_model()
    
    # Load sample images from MIMIC-CXR
    print("\nLoading sample images from MIMIC-CXR...")
    sample_images = load_sample_images(mimic_cxr_dir, num_samples=3)
    
    if not sample_images:
        print("No sample images found. Please check your MIMIC-CXR path.")
        return
    
    print(f"Loaded {len(sample_images)} sample images")
    
    # Test the model on sample images
    test_gemini_on_samples(model, sample_images)
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main()
