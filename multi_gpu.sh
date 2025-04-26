#!/bin/bash

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs

# Print GPU info
echo "=== GPU Information ==="
nvidia-smi

# Create necessary directories
mkdir -p data/MMCaD
mkdir -p data/train data/val data/test

# Step 1: Download MIMIC-IV data if needed
read -p "Do you want to download MIMIC-IV hospital data? (y/n): " download_mimiciv
if [ "$download_mimiciv" == "y" ]; then
    echo "Please edit download_mimiciv.sh with your PhysioNet username first."
    read -p "Have you edited the script with your username? (y/n): " edited
    if [ "$edited" == "y" ]; then
        chmod +x download_mimiciv.sh
        ./download_mimiciv.sh
    else
        echo "Please edit the script and try again."
        exit 1
    fi
fi

# Step 2: Create dataset indexes
echo "=== Creating Dataset Indexes ==="
python create_indexes.py

# Step 3: Run data preparation
echo "=== Running Data Preparation ==="
python prepare_data_modified.py

# Optionally run discharge summary embedding generation
read -p "Do you want to generate discharge summary embeddings? This requires GPU and may take a long time. (y/n): " generate_embeddings
if [ "$generate_embeddings" == "y" ]; then
    echo "=== Generating Discharge Summary Embeddings ==="
    python -c "from prepare_data_modified import prepare_discharge_summary_embeddings; prepare_discharge_summary_embeddings('data/train_idx.json', 'train')"
    python -c "from prepare_data_modified import prepare_discharge_summary_embeddings; prepare_discharge_summary_embeddings('data/val_idx.json', 'val')"
    python -c "from prepare_data_modified import prepare_discharge_summary_embeddings; prepare_discharge_summary_embeddings('data/test_idx.json', 'test')"
fi

echo "=== Processing Complete ==="
