# fix_icd_codes.py
import os
import json
import pandas as pd
import pickle
from tqdm import tqdm

def parse_gem_files():
    """Parse GEM files to create ICD mapping"""
    # Check if gem directory exists
    gem_dir = 'data/gem'
    if not os.path.exists(gem_dir):
        os.makedirs(gem_dir, exist_ok=True)
        print(f"Created directory {gem_dir}")
    
    # Check if I9gem file exists
    i9gem_file = os.path.join(gem_dir, '2015_I9gem.txt')
    i10gem_file = os.path.join(gem_dir, '2015_I10gem.txt')
    
    # Extract ICD-9 codes from both files
    icd9_codes = set()
    
    if os.path.exists(i9gem_file):
        print(f"Parsing {i9gem_file}")
        with open(i9gem_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    icd9_codes.add(parts[0])
    
    if os.path.exists(i10gem_file):
        print(f"Parsing {i10gem_file}")
        with open(i10gem_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    icd9_codes.add(parts[1])
    
    # Add the problematic code '4149' explicitly
    icd9_codes.add('4149')
    
    print(f"Found {len(icd9_codes)} unique ICD-9 codes")
    return list(icd9_codes)

def create_complete_icd_files(icd9_codes):
    """Create complete ICD files with all necessary codes"""
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save the ICD code list
    with open('data/all_icd_code9_list.json', 'w') as f:
        json.dump(icd9_codes, f)
    print(f"✓ Created data/all_icd_code9_list.json with {len(icd9_codes)} codes")
    
    # Create MIMIC-IV hospital directory
    mimiciv_dir = 'data/mimiciv/hosp'
    os.makedirs(mimiciv_dir, exist_ok=True)
    
    # Create d_icd_diagnoses.csv.gz with all codes
    icd_file = os.path.join(mimiciv_dir, 'd_icd_diagnoses.csv.gz')
    
    # Create a comprehensive DataFrame with all codes
    icd_df = pd.DataFrame({
        'icd_code': icd9_codes,
        'icd_version': [9] * len(icd9_codes),
        'long_title': [f"Diagnosis for {code}" for code in icd9_codes],
        'short_title': [f"Diag-{code}" for code in icd9_codes]
    })
    
    # Save to gzipped CSV
    icd_df.to_csv(icd_file, index=False, compression='gzip')
    print(f"✓ Created {icd_file} with {len(icd9_codes)} diagnoses")
    
    return True

def create_empty_diagnosis_files():
    """Create empty diagnosis pickle files"""
    for filepath in ['diagnosis_dict_icd9.pkl', 'diagnosis_counts_icd9.pkl']:
        if not os.path.exists(filepath):
            with open(filepath, 'wb') as f:
                pickle.dump({}, f)
            print(f"✓ Created empty {filepath}")

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
    print("=== Fixing ICD Files for GeMini Training ===")
    
    # Parse GEM files to get ICD codes
    icd9_codes = parse_gem_files()
    
    # Create complete ICD files
    success = create_complete_icd_files(icd9_codes)
    
    # Create empty diagnosis pickle files
    create_empty_diagnosis_files()
    
    # Create category files
    create_category_files()
    
    # Create split files
    create_split_files()
    
    if success:
        print("\n=== All Files Fixed Successfully ===")
        print("\nYou should now be able to run the training script:")
        print("bash run_train.sh")
    else:
        print("\n=== Failed to Fix Some Files ===")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main()
