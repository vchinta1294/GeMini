# xray_multimodal.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel, BertTokenizer, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class MultimodalXrayModel(nn.Module):
    """
    A simplified multimodal model for chest X-ray analysis
    Combines a ViT for image encoding and BiomedBERT for text encoding
    """
    def __init__(self, num_classes=14, fusion_dim=512):
        super().__init__()
        
        # Load image encoder (ViT)
        self.image_encoder = ViTModel.from_pretrained("models/vit-base-patch16-224-in21k")
        
        # Load text encoder (BiomedBERT)
        self.text_encoder = BertModel.from_pretrained("models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
        
        # Get encoder output dimensions
        self.img_dim = self.image_encoder.config.hidden_size  # 768 for ViT base
        self.text_dim = self.text_encoder.config.hidden_size  # 768 for BERT base
        
        # Projection layers
        self.img_projector = nn.Sequential(
            nn.Linear(self.img_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU()
        )
        
        # Fusion mechanism (simple addition with attention)
        self.fusion_attention = nn.Sequential(
            nn.Linear(fusion_dim * 2, 2),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, num_classes)
        )
    
    def forward(self, img_inputs, text_inputs=None):
        """
        Forward pass with image and optional text inputs
        
        Args:
            img_inputs: Dict with 'pixel_values' tensor of shape [batch_size, 3, 224, 224]
            text_inputs: Dict with 'input_ids' and 'attention_mask' tensors
        
        Returns:
            Dict with 'logits' and other outputs
        """
        # Process image
        img_outputs = self.image_encoder(**img_inputs)
        img_features = img_outputs.pooler_output  # [batch_size, 768]
        img_projected = self.img_projector(img_features)  # [batch_size, fusion_dim]
        
        # Process text if available
        if text_inputs is not None:
            text_outputs = self.text_encoder(**text_inputs)
            text_features = text_outputs.pooler_output  # [batch_size, 768]
            text_projected = self.text_projector(text_features)  # [batch_size, fusion_dim]
            
            # Combine features with attention weights
            combined = torch.cat([img_projected, text_projected], dim=1)  # [batch_size, fusion_dim*2]
            attention_weights = self.fusion_attention(combined)  # [batch_size, 2]
            
            # Apply attention weights
            fused_features = (
                attention_weights[:, 0].unsqueeze(1) * img_projected + 
                attention_weights[:, 1].unsqueeze(1) * text_projected
            )
        else:
            # Use only image features if text is not available
            fused_features = img_projected
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'features': fused_features
        }

class CheXpertDataset(Dataset):
    """
    Dataset for CheXpert labels from MIMIC-CXR with optional text data
    """
    def __init__(self, mimic_cxr_dir, split='train', use_text=True, max_samples=None):
        """
        Args:
            mimic_cxr_dir: Path to MIMIC-CXR-JPG dataset
            split: 'train', 'val', or 'test'
            use_text: Whether to include text data
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.mimic_cxr_dir = mimic_cxr_dir
        self.use_text = use_text
        
        # Load split file
        split_file = os.path.join(mimic_cxr_dir, 'mimic-cxr-2.0.0-split.csv.gz')
        self.split_df = pd.read_csv(split_file, compression='gzip')
        
        # Filter by split
        if split == 'train':
            self.split_df = self.split_df[self.split_df['split'] == 'train']
        elif split == 'val':
            self.split_df = self.split_df[self.split_df['split'] == 'validate']
        elif split == 'test':
            self.split_df = self.split_df[self.split_df['split'] == 'test']
        
        # Load CheXpert labels
        chexpert_file = os.path.join(mimic_cxr_dir, 'mimic-cxr-2.0.0-chexpert.csv.gz')
        self.chexpert_df = pd.read_csv(chexpert_file, compression='gzip')
        
        # Merge split and chexpert data
        self.df = pd.merge(self.split_df, self.chexpert_df, on=['subject_id', 'study_id'], how='inner')
        
        # Define the 14 CheXpert classes
        self.label_columns = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # Process labels: convert -1 (uncertain) to 0 (negative) and fill NaN with 0
        for col in self.label_columns:
            self.df[col] = self.df[col].fillna(0)
            self.df[col] = self.df[col].replace(-1, 0)
        
        # Limit number of samples for debugging
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.iloc[:max_samples]
        
        print(f"Loaded {len(self.df)} samples for {split} split")
        
        # Load processors
        self.vit_processor = ViTImageProcessor.from_pretrained("models/vit-base-patch16-224-in21k")
        self.tokenizer = BertTokenizer.from_pretrained("models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get image path
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        
        prefix = f"p{str(subject_id)[:2]}"
        img_path = os.path.join(
            self.mimic_cxr_dir, 
            'files',
            prefix, 
            f"p{subject_id}", 
            f"s{study_id}", 
            f"{dicom_id}.jpg"
        )
        
        # Load and process image
        try:
            image = Image.open(img_path).convert('RGB')
            img_inputs = self.vit_processor(images=image, return_tensors="pt")
            # Remove batch dimension
            img_inputs = {k: v.squeeze(0) for k, v in img_inputs.items()}
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            img_inputs = self.vit_processor(images=Image.new('RGB', (224, 224), color=0), return_tensors="pt")
            img_inputs = {k: v.squeeze(0) for k, v in img_inputs.items()}
        
        # Get labels
        labels = torch.tensor(row[self.label_columns].values.astype(float), dtype=torch.float32)
        
        # Process text if available and requested
        text_inputs = None
        if self.use_text:
            # For text, we'll use view position and other metadata as a simple example
            # In a real application, you might want to use radiology reports
            view_position = row.get('ViewPosition', 'Unknown')
            text = f"Chest X-ray with view position {view_position}"
            
            text_inputs = self.tokenizer(
                text, 
                padding='max_length', 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            # Remove batch dimension
            text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}
        
        return {
            'img_inputs': img_inputs,
            'text_inputs': text_inputs,
            'labels': labels,
            'subject_id': subject_id,
            'study_id': study_id,
            'dicom_id': dicom_id
        }

def train_model(model, train_loader, val_loader, num_epochs=5, lr=1e-4, device='cuda'):
    """
    Train the multimodal model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
    
    Returns:
        The trained model
    """
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in progress_bar:
            # Move batch to device
            img_inputs = {k: v.to(device) for k, v in batch['img_inputs'].items()}
            text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()} if batch['text_inputs'] is not None else None
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(img_inputs, text_inputs)
            logits = outputs['logits']
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                # Move batch to device
                img_inputs = {k: v.to(device) for k, v in batch['img_inputs'].items()}
                text_inputs = {k: v.to(device) for k, v in batch['text_inputs'].items()} if batch['text_inputs'] is not None else None
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(img_inputs, text_inputs)
                logits = outputs['logits']
                
                # Compute loss
                loss = criterion(logits, labels)
                
                # Update metrics
                val_loss += loss.item()
                
                # Save predictions and labels for metrics
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
        
        # Compute metrics
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Compute AUC for each class
        from sklearn.metrics import roc_auc_score
        aucs = []
        for i in range(all_labels.shape[1]):
            if np.sum(all_labels[:, i] == 1) > 0 and np.sum(all_labels[:, i] == 0) > 0:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
        
        mean_auc = np.mean(aucs) if aucs else 0.0
        
        # Print epoch metrics
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val AUC: {mean_auc:.4f}")
    
    return model

def main():
    # Set parameters
    mimic_cxr_dir = '/home/vchinta/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0'
    batch_size = 32
    num_epochs = 5
    max_samples = 1000  # Limit for testing, set to None for full dataset
    use_text = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create datasets
    train_dataset = CheXpertDataset(mimic_cxr_dir, 'train', use_text, max_samples)
    val_dataset = CheXpertDataset(mimic_cxr_dir, 'val', use_text, max_samples//10)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = MultimodalXrayModel(num_classes=14, fusion_dim=512)
    
    # Train model
    model = train_model(model, train_loader, val_loader, num_epochs, device=device)
    
    # Save model
    torch.save(model.state_dict(), 'xray_multimodal_model.pth')
    print("Model saved to xray_multimodal_model.pth")

if __name__ == "__main__":
    main()
