# update_config.py
import json
import os

def update_config():
    """Update the config.json file with local paths"""
    config_path = "output/GEMINI/pretrained/config.json"
    
    # Check if the file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return False
    
    # Read the current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Save a backup
    backup_path = f"{config_path}.bak"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created backup at {backup_path}")
    
    # Update paths to point to local directories
    if '_name_or_path' in config:
        old_path = config['_name_or_path']
        new_path = "models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        config['_name_or_path'] = new_path
        print(f"Updated _name_or_path: {old_path} -> {new_path}")
    
    if 'text_model_path' in config:
        old_path = config['text_model_path']
        new_path = "models/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        config['text_model_path'] = new_path
        print(f"Updated text_model_path: {old_path} -> {new_path}")
    
    if 'img_model_path16' in config:
        old_path = config['img_model_path16']
        new_path = "models/vit-base-patch16-224-in21k"
        config['img_model_path16'] = new_path
        print(f"Updated img_model_path16: {old_path} -> {new_path}")
    
    if 'img_model_path32' in config:
        if config['img_model_path32'] != "models/vit-base-patch32-224-in21k":
            old_path = config['img_model_path32'] 
            new_path = "models/vit-base-patch32-224-in21k"
            config['img_model_path32'] = new_path
            print(f"Updated img_model_path32: {old_path} -> {new_path}")
    
    # Save the updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Updated config saved to {config_path}")
    return True

if __name__ == "__main__":
    print("=== Updating Config Paths ===")
    success = update_config()
    
    if success:
        print("\n=== Paths Updated Successfully ===")
        print("\nYou should now be able to run the test script:")
        print("python test_script.py")
    else:
        print("\n=== Failed to Update Paths ===")
