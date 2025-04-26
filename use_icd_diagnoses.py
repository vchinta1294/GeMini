# use_icd_diagnoses.py
import os
import shutil
import pandas as pd
import pickle
import json
from tqdm import tqdm

def patch_dataset_file():
    """Patch the dataset.py file to handle the missing '4149' code"""
    dataset_path = 'src/dataset.py'
    
    # First, create a backup
    backup_path = f"{dataset_path}.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(dataset_path, backup_path)
        print(f"✓ Created backup of {dataset_path} at {backup_path}")
    
    # Find the line that tries to delete '4149'
    target_line = "        del self.diagnosis_label_ids['4149']"
    replacement_line = "        if '4149' in self.diagnosis_label_ids:\n            del self.diagnosis_label_ids['4149']"
    
    # Read the file content
    with open(dataset_path, 'r') as f:
        content = f.read()
    
    # Replace the line
    if target_line in content:
        modified_content = content.replace(target_line, replacement_line)
        
        # Write the modified content back
        with open(dataset_path, 'w') as f:
            f.write(modified_content)
        
        print(f"✓ Successfully patched {dataset_path}")
        return True
    else:
        print(f"✗ Could not find the target line in {dataset_path}")
        return False

def create_mimiciv_dir_from_csv():
    """Create the mimiciv hosp directory and copy the d_icd_diagnoses.csv file"""
    mimiciv_dir = 'data/mimiciv/hosp'
    os.makedirs(mimiciv_dir, exist_ok=True)
    
    source_csv = 'd_icd_diagnoses.csv'
    target_csv_gz = os.path.join(mimiciv_dir, 'd_icd_diagnoses.csv.gz')
    
    if os.path.exists(source_csv):
        print(f"Reading {source_csv}...")
        icd_df = pd.read_csv(source_csv)
        
        # Save to gzipped CSV
        print(f"Saving to {target_csv_gz}...")
        icd_df.to_csv(target_csv_gz, index=False, compression='gzip')
        
        print(f"✓ Created {target_csv_gz} with {len(icd_df)} diagnoses")
        
        # Extract ICD-9 codes
        icd9_codes = icd_df[icd_df['icd_version'] == 9]['icd_code'].astype(str).tolist()
        print(f"Extracted {len(icd9_codes)} ICD-9 codes")
        
        # Save the ICD code list
        with open('data/all_icd_code9_list.json', 'w') as f:
            json.dump(icd9_codes, f)
        print(f"✓ Created data/all_icd_code9_list.json with {len(icd9_codes)} codes")
        
        return True
    else:
        print(f"✗ Could not find {source_csv}")
        return False

def create_category_files():
    """Create required category mapping files"""
    category_files = [
        ('micro_spec_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_test_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_org_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_ab_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_dilution_comparison_category_ids.json', {"0": 0, "-100.0": 1}),
        ('patient_category_ids.json', {"M": 0, "F": 1, "WHITE": 2, "BLACK": 3, "ASIAN": 4, "HISPANIC": 5, "OTHER": 6, "AMBULANCE": 7, "WALK": 8, "TAXI": 9, "-100.0": 10}),
        ('triage_category_ids.json', {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "-100.0": 6})
    ]
    
    for filename, data in category_files:
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump(data, f)
            print(f"✓ Created {filepath}")

def create_split_files():
    """Create diagnosis list files for each split"""
    for split in ['train', 'val', 'test']:
        filepath = f'all_diagnosis_list_{split}.json'
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump(list(range(30)), f)
            print(f"✓ Created {filepath}")

def main():
    print("=== Setting Up ICD Files for GeMini Training ===")
    
    # Create mimiciv directory from CSV
    csv_success = create_mimiciv_dir_from_csv()
    
    # Patch dataset file
    patch_success = patch_dataset_file()
    
    # Create category files
    create_category_files()
    
    # Create split files
    create_split_files()
    
    if csv_success and patch_success:
        print("\n=== Setup Completed Successfully ===")
        print("\nYou should now be able to run the training script:")
        print("bash run_train.sh")
    else:
        print("\n=== Setup Completed with Warnings ===")
        print("Please check the error messages above before running the training script.")

if __name__ == "__main__":
    main()
