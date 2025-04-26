# check_model.py
import os
import torch
from src.models import GEMINI_scratch
from transformers import PretrainedConfig

def check_model():
    """Check if the model can be loaded properly"""
    model_path = "output/GEMINI/pretrained"
    
    try:
        # Load config
        config = PretrainedConfig.from_pretrained(model_path)
        print("Config loaded successfully")
        
        # Initialize model with config
        model = GEMINI_scratch(config)
        print("Model initialized successfully")
        
        # Load weights
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        print(f"State dict loaded with {len(state_dict)} keys")
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params} parameters")
        
        return True
    except Exception as e:
        print(f"Error checking model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Checking GeMini Model ===")
    success = check_model()
    
    if success:
        print("\n=== Model Check Completed Successfully ===")
    else:
        print("\n=== Model Check Failed ===")
