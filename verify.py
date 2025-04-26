# verify_improved.py
import os
import torch
import json
import pandas as pd
from transformers import BertTokenizer, ViTImageProcessor

def create_image_paths_file():
    """Create the missing unique_image_paths.json file from MIMIC-CXR metadata"""
    mimic_cxr_dir = '/home/vchinta/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0'
    output_path = 'data/unique_image_paths.json'
    
    if os.path.exists(output_path):
        print(f"✓ {output_path} already exists")
        return
    
    print(f"Creating {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load metadata
    metadata_file = os.path.join(mimic_cxr_dir, 'mimic-cxr-2.0.0-metadata.csv.gz')
    split_file = os.path.join(mimic_cxr_dir, 'mimic-cxr-2.0.0-split.csv.gz')
    
    # First, let's inspect the metadata files to understand their structure
    metadata_df = pd.read_csv(metadata_file, compression='gzip')
    split_df = pd.read_csv(split_file, compression='gzip')
    
    print("Metadata columns:", metadata_df.columns.tolist())
    print("Split columns:", split_df.columns.tolist())
    
    # Create image paths using the correct columns
    image_paths = []
    
    # Use split_df since it has subject_id and study_id
    for _, row in split_df.iterrows():
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        
        # Pattern is: p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        prefix = f"p{str(subject_id)[:2]}"
        path = f"{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        image_paths.append(path)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(image_paths, f)
    
    print(f"✓ Created {output_path} with {len(image_paths)} image paths")
    return image_paths

def verify_model_paths():
    """Verify model paths and download if needed"""
    model_paths = {
        "text_model": "models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "img_model16": "models/vit-base-patch16-224-in21k",
        "img_model32": "models/vit-base-patch32-224-in21k"
    }
    
    os.makedirs("models", exist_ok=True)
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"✓ {name} exists at {path}")
        else:
            print(f"✗ {name} missing at {path}")
            print(f"  Please download {name} to {path}")

def verify_data_paths():
    """Verify required data paths"""
    required_paths = [
        'data/train_idx.json',
        'data/val_idx.json',
        'data/test_idx.json',
        'data/data_identifiers.json',
        'data/mimiciv/hosp/d_labitems.csv.gz'
    ]
    
    for path in required_paths:
        if os.path.exists(path):
            print(f"✓ {path} exists")
        else:
            print(f"✗ {path} is missing")

def create_data_dirs():
    """Create necessary data directories"""
    dirs = [
        "data/MMCaD",
        "data/train",
        "data/val",
        "data/test",
        "output/GEMINI"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory {dir_path}")

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Check if there's a compatibility issue
        cuda_capability = torch.cuda.get_device_capability()
        print(f"  CUDA capability: {cuda_capability}")
        if cuda_capability[0] >= 8:
            print("⚠️ Your GPU has CUDA capability >= 8.0. You might need to upgrade PyTorch")
            print("  for optimal compatibility. See instructions for upgrading PyTorch.")
    else:
        print("✗ No GPU available. This model requires GPU for training.")

def download_models():
    """Download required model checkpoints"""
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Define model paths
    models = {
        "BiomedNLP-BiomedBERT": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "ViT-16": "google/vit-base-patch16-224-in21k",
        "ViT-32": "google/vit-base-patch32-224-in21k"
    }
    
    # Download models
    for name, model_id in models.items():
        print(f"Downloading {name}...")
        
        if "BiomedNLP" in model_id:
            save_dir = "models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
            if not os.path.exists(save_dir):
                tokenizer = BertTokenizer.from_pretrained(model_id)
                tokenizer.save_pretrained(save_dir)
                print(f"✓ Downloaded {name} tokenizer")
            else:
                print(f"✓ {name} tokenizer already exists")
        
        if "vit-base-patch16" in model_id:
            save_dir = "models/vit-base-patch16-224-in21k"
            if not os.path.exists(save_dir):
                processor = ViTImageProcessor.from_pretrained(model_id)
                processor.save_pretrained(save_dir)
                print(f"✓ Downloaded {name} processor")
            else:
                print(f"✓ {name} processor already exists")
        
        if "vit-base-patch32" in model_id:
            save_dir = "models/vit-base-patch32-224-in21k"
            if not os.path.exists(save_dir):
                processor = ViTImageProcessor.from_pretrained(model_id)
                processor.save_pretrained(save_dir)
                print(f"✓ Downloaded {name} processor")
            else:
                print(f"✓ {name} processor already exists")
    
    print("All models downloaded successfully!")

def main():
    print("=== GeMini Setup Verification ===")
    
    # Check GPU
    print("\n=== GPU Verification ===")
    check_gpu()
    
    # Create data directories
    print("\n=== Creating Data Directories ===")
    create_data_dirs()
    
    # Verify data paths
    print("\n=== Verifying Data Paths ===")
    verify_data_paths()
    
    # Verify model paths
    print("\n=== Verifying Model Paths ===")
    verify_model_paths()
    
    # Create unique_image_paths.json file
    print("\n=== Creating Image Paths File ===")
    image_paths = create_image_paths_file()
    
    # Download models
    print("\n=== Downloading Models ===")
    download_models()
    
    # Verify image paths
    if image_paths:
        print("\n=== Verifying Sample Image Paths ===")
        cxr_dir = '/home/vchinta/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0/files'
        sample_paths = image_paths[:5]
        for path in sample_paths:
            full_path = os.path.join(cxr_dir, path)
            if os.path.exists(full_path):
                print(f"✓ {path} exists")
            else:
                print(f"✗ {path} is missing")
    
    print("\n=== Setup Verification Complete ===")
    print("\nNext steps:")
    print("1. Run the training script with `bash run_train.sh`")

if __name__ == "__main__":
    main()
