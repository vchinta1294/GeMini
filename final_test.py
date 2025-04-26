# final_test_fixed.py
import os
import torch
import json
from PIL import Image
from transformers import BertTokenizer, ViTImageProcessor, PretrainedConfig
import matplotlib.pyplot as plt
import random

# Import the model class
from src.models import GEMINI_scratch

def load_model():
    """Load the pretrained GeMini model"""
    model_path = "output/GEMINI/pretrained"
    
    try:
        # Load config
        config = PretrainedConfig.from_pretrained(model_path)
        
        # Initialize model with config
        model = GEMINI_scratch(config)
        
        # Load weights
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        
        # Filter out missing keys
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
        
        # Load the filtered state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def load_image(path):
    """Load and process an image for the model"""
    try:
        image = Image.open(path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def test_model():
    """Test the GeMini model on a sample image"""
    # Load model
    model = load_model()
    if model is None:
        return False
    
    # Set model to evaluation mode
    model.eval()
    
    # Load tokenizer and image processors
    tokenizer = BertTokenizer.from_pretrained("models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    vit_processor16 = ViTImageProcessor.from_pretrained("models/vit-base-patch16-224-in21k")
    vit_processor32 = ViTImageProcessor.from_pretrained("models/vit-base-patch32-224-in21k")
    
    # Find a sample image from MIMIC-CXR
    mimic_cxr_dir = '/home/vchinta/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
    
    # Get all p* directories
    p_dirs = [d for d in os.listdir(mimic_cxr_dir) if d.startswith('p') and os.path.isdir(os.path.join(mimic_cxr_dir, d))]
    
    sample_image_path = None
    
    # Find an image
    for p_dir in p_dirs:
        p_path = os.path.join(mimic_cxr_dir, p_dir)
        patient_dirs = [d for d in os.listdir(p_path) if os.path.isdir(os.path.join(p_path, d))]
        
        for patient_dir in patient_dirs:
            patient_path = os.path.join(p_path, patient_dir)
            study_dirs = [d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]
            
            for study_dir in study_dirs:
                study_path = os.path.join(patient_path, study_dir)
                image_files = [f for f in os.listdir(study_path) if f.endswith('.jpg')]
                
                if image_files:
                    sample_image_path = os.path.join(study_path, image_files[0])
                    break
            
            if sample_image_path:
                break
        
        if sample_image_path:
            break
    
    if not sample_image_path:
        print("No sample image found")
        return False
    
    print(f"Using sample image: {sample_image_path}")
    
    # Load and process the image
    image = load_image(sample_image_path)
    if image is None:
        return False
    
    # Process with ViT processors
    processed16 = vit_processor16(images=image, return_tensors="pt")
    processed32 = vit_processor32(images=image, return_tensors="pt")
    
    # Prepare a batch of 4 images (as expected by the model)
    # The first is the actual image, the rest are zeros (placeholders)
    image_feature = torch.zeros(1, 4, 3, 224, 224)
    image_feature[0, 0] = processed16['pixel_values'][0]  # First image is processed with ViT-16
    
    # Create dummy text input
    text = "This is a chest X-ray image."
    text_inputs = tokenizer(text, return_tensors="pt")
    
    # Save the image for reference
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title("Sample X-ray Image")
    plt.axis('off')
    plt.savefig("sample_xray.png")
    plt.close()
    print(f"Sample image saved as sample_xray.png")
    
    # Create dummy inputs for all the other parameters based on model's expected shapes
    lab_inputs = {
        'labevent_number_input': torch.zeros(1, 100, 2000),
        'labevent_category_input': torch.zeros(1, 100, dtype=torch.long)
    }
    
    micro_inputs = {
        'microbiology_category_input': torch.zeros(1, 75, dtype=torch.long),
        'microbiology_number_input': torch.zeros(1, 15, 2000)
    }
    
    patient_inputs = {
        'patient_category_input': torch.zeros(1, 3, dtype=torch.long),
        'patient_number_input': torch.zeros(1, 1, 2000)
    }
    
    triage_inputs = {
        'triage_category_input': torch.zeros(1, 2, dtype=torch.long),
        'triage_number_input': torch.zeros(1, 6, 2000)
    }
    
    # CRITICAL: These shapes need to match exactly what the model expects
    # Based on the error, microbiology_comment_embeddings should be [b*num_comments, max_tokens]
    b = 1
    num_comments = 15
    max_tokens = 512
    
    # Create dummy embeddings
    med_history_tokens = tokenizer("Medical history: None.", return_tensors="pt", padding="max_length", truncation=True, max_length=277)
    fam_history_tokens = tokenizer("Family history: None.", return_tensors="pt", padding="max_length", truncation=True, max_length=61)
    chief_complaint_tokens = tokenizer("Chest pain", return_tensors="pt", padding="max_length", truncation=True, max_length=22)
    micro_comment_tokens = []
    
    for _ in range(num_comments):
        tokens = tokenizer("No infection.", return_tensors="pt", padding="max_length", truncation=True, max_length=max_tokens)
        micro_comment_tokens.append(tokens['input_ids'])
    
    micro_comment_input_ids = torch.cat(micro_comment_tokens, dim=0)
    micro_comment_attention_mask = torch.ones(num_comments, max_tokens)
    
    text_embeddings = {
        'medical_history_embeddings': med_history_tokens['input_ids'],
        'medical_history_attention_mask': med_history_tokens['attention_mask'],
        'family_history_embeddings': fam_history_tokens['input_ids'],
        'family_history_attention_mask': fam_history_tokens['attention_mask'],
        'chiefcomplaint_embedding': chief_complaint_tokens['input_ids'],
        'chiefcomplaint_attention_mask': chief_complaint_tokens['attention_mask'],
        'microbiology_comment_embeddings': micro_comment_input_ids,
        'microbiology_comment_attention_mask': micro_comment_attention_mask
    }
    
    # Calculate total sequence length for attention mask
    img_tokens = 347
    lab_tokens = 200
    micro_tokens = 105
    med_history_tokens = 277
    fam_history_tokens = 61
    patient_tokens = 4
    triage_tokens = 30
    chief_complaint_tokens = 22
    total_seq_len = img_tokens + lab_tokens + micro_tokens + med_history_tokens + fam_history_tokens + patient_tokens + triage_tokens + chief_complaint_tokens
    
    # Create attention masks
    mask_inputs = {
        'total_attention_mask': torch.ones(1, total_seq_len),
        'multimodal_input_type': torch.zeros(1, total_seq_len, dtype=torch.long)
    }
    
    # Assign different token types
    token_idx = 0
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+img_tokens] = 0  # Image
    token_idx += img_tokens
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+lab_tokens] = 1  # Lab
    token_idx += lab_tokens
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+micro_tokens] = 2  # Micro
    token_idx += micro_tokens
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+med_history_tokens] = 3  # Med history
    token_idx += med_history_tokens
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+fam_history_tokens] = 4  # Family history
    token_idx += fam_history_tokens
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+patient_tokens] = 5  # Patient
    token_idx += patient_tokens
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+triage_tokens] = 5  # Triage
    token_idx += triage_tokens
    mask_inputs['multimodal_input_type'][0, token_idx:token_idx+chief_complaint_tokens] = 6  # Chief complaint
    
    # Run inference
    try:
        print("Running inference...")
        with torch.no_grad():
            # Use the expected parameter names
            outputs = model(
                image_feature=image_feature,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                labevent_number_input=lab_inputs['labevent_number_input'],
                labevent_category_input=lab_inputs['labevent_category_input'],
                microbiology_category_input=micro_inputs['microbiology_category_input'],
                microbiology_number_input=micro_inputs['microbiology_number_input'],
                microbiology_comment_embeddings=text_embeddings['microbiology_comment_embeddings'],
                microbiology_comment_attention_mask=text_embeddings['microbiology_comment_attention_mask'],
                medical_history_embeddings=text_embeddings['medical_history_embeddings'],
                medical_history_attention_mask=text_embeddings['medical_history_attention_mask'],
                family_history_embeddings=text_embeddings['family_history_embeddings'],
                family_history_attention_mask=text_embeddings['family_history_attention_mask'],
                patient_category_input=patient_inputs['patient_category_input'],
                patient_number_input=patient_inputs['patient_number_input'],
                triage_category_input=triage_inputs['triage_category_input'],
                triage_number_input=triage_inputs['triage_number_input'],
                chiefcomplaint_embedding=text_embeddings['chiefcomplaint_embedding'],
                chiefcomplaint_attention_mask=text_embeddings['chiefcomplaint_attention_mask'],
                total_attention_mask=mask_inputs['total_attention_mask'],
                multimodal_input_type=mask_inputs['multimodal_input_type']
            )
        
        # Print outputs
        print("Output keys:", outputs.keys())
        
        if 'hidden_states' in outputs:
            print(f"Hidden states shape: {outputs['hidden_states'].shape}")
        
        if 'pooler_output' in outputs:
            print(f"Pooler output shape: {outputs['pooler_output'].shape}")
        
        if 'logits' in outputs:
            print(f"Logits shape: {outputs['logits'].shape}")
        
        print("Inference completed successfully")
        return True
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Testing GeMini Model ===")
    success = test_model()
    
    if success:
        print("\n=== Test Completed Successfully ===")
    else:
        print("\n=== Test Failed ===")
