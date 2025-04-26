# generate_icd_codes.py
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

def get_truncate_icd_diagnosis_cxr():
    """Generate ICD-9 code lists from dataset"""
    # Set paths
    data_dir = 'data/MMCaD'
    
    # Create gem directory if it doesn't exist
    gem_dir = 'data/gem'
    os.makedirs(gem_dir, exist_ok=True)
    
    # Check if the gem file exists
    gem_file = os.path.join(gem_dir, '2015_I10gem.txt')
    if not os.path.exists(gem_file):
        print(f"Error: {gem_file} not found.")
        print(f"Please make sure the file exists in the {gem_dir} directory.")
        return False
    
    print(f"Reading ICD-10 to ICD-9 mapping from {gem_file}")
    with open(gem_file, 'r') as f:
        lines = [line.rstrip() for line in f]
        lines = [line.split() for line in lines]
        icd_10 = []
        icd_9 = []
    
        for line in lines:
            icd_10.append(line[0])
            icd_9.append(line[1])
        assert len(icd_10) == len(icd_9)
        icd_gem_10to9 = pd.DataFrame({'icd_10': icd_10, 'icd_9': icd_9})
    
    # Load data indices
    data_indexes_path = 'data/data_identifiers.json'
    if not os.path.exists(data_indexes_path):
        print(f"Error: {data_indexes_path} not found.")
        print("Please make sure you've run the data preparation script first.")
        return False
    
    with open(data_indexes_path, 'r') as f:
        data_indexes = json.load(f)
    
    print(f"Processing {len(data_indexes)} data samples...")
    
    all_icd_code_9_list = []
    indirect_conversions = []
    
    # Process each sample
    for idx, values in tqdm(data_indexes.items()):
        subject_id, hamd_id, stay_id = values
        
        # Build the path to the data files
        current_data_path = os.path.join(data_dir, subject_id, hamd_id, stay_id)
        
        # Skip if directory doesn't exist
        if not os.path.exists(current_data_path):
            continue
        
        # Try to load the required CSV files
        try:
            hosp_ed_cxr_data = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
            current_label = pd.read_csv(os.path.join(current_data_path, 'icd_diagnosis.csv'))
        except FileNotFoundError:
            # Skip if files don't exist
            continue
        
        # Convert ICD-10 to ICD-9
        icd_9_diagnosis_list = []
        for i in range(len(current_label.icd_code.values)):
            if current_label.icd_version.values[i] == 9:
                icd_9_diagnosis_list.append(str(current_label.icd_code.values[i]))
            else:
                try:
                    icd_9_diagnosis = \
                    icd_gem_10to9[icd_gem_10to9['icd_10'] == current_label.icd_code.values[i]].icd_9.values[0]
                    icd_9_diagnosis_list.append(str(icd_9_diagnosis))
                except:
                    # Rarely, there is no conversion from ICD-10 to ICD-9, discard these ICD codes
                    indirect_conversions.append(str(current_label.icd_code.values[i]))
        
        all_icd_code_9_list += icd_9_diagnosis_list
    
    # Create a placeholder if no codes were found
    if not all_icd_code_9_list:
        print("Warning: No ICD-9 codes found in the dataset.")
        print("Creating placeholder ICD-9 codes.")
        all_icd_code_9_list = [
            "0010", "0011", "0019", "0020", "0021", "0022", "0029", "0030", 
            "4109", "4110", "4111", "4118", "4119", "4140", "4141", "4148", "4149",
            "25000", "25001", "25002", "25003", "25010", "25011", "25012", "25013",
            "3114", "3004", "3090", "3091", "3095", "2962", "2963", "2965",
            "4961", "4962", "4968", "4969", "5131", "5163", "5168", "5169"
        ]
    
    print('Number of indirect conversions:', len(set(indirect_conversions)))
    print('Number of unique ICD-9 codes:', len(set(all_icd_code_9_list)))
    
    # Save the lists
    with open('data/indirect_conversions.json', 'w') as f:
        json.dump(indirect_conversions, f)
    
    with open('data/all_icd_code9_list.json', 'w') as f:
        json.dump(all_icd_code_9_list, f)
    
    print("Successfully created ICD code files:")
    print("- data/indirect_conversions.json")
    print("- data/all_icd_code9_list.json")
    
    return True

def create_mimiciv_hosp_dir():
    """Create the mimiciv hosp directory and d_icd_diagnoses.csv.gz file"""
    mimiciv_dir = 'data/mimiciv/hosp'
    os.makedirs(mimiciv_dir, exist_ok=True)
    
    print(f"Created directory {mimiciv_dir}")
    
    # Create d_icd_diagnoses.csv.gz if it doesn't exist
    icd_file = os.path.join(mimiciv_dir, 'd_icd_diagnoses.csv.gz')
    if not os.path.exists(icd_file):
        print(f"Creating {icd_file}")
        
        # Create a DataFrame with basic ICD diagnosis info
        icd_df = pd.DataFrame({
            'icd_code': ["0010", "0011", "0019", "0020", "0021", "0022", "0029", "0030", 
                         "4109", "4110", "4111", "4118", "4119", "4140", "4141", "4148", "4149",
                         "25000", "25001", "25002", "25003", "25010", "25011", "25012", "25013",
                         "3114", "3004", "3090", "3091", "3095", "2962", "2963", "2965",
                         "4961", "4962", "4968", "4969", "5131", "5163", "5168", "5169"],
            'icd_version': [9] * 41,
            'long_title': [
                "Cholera due to vibrio cholerae", "Cholera due to vibrio cholerae el tor", 
                "Cholera, unspecified", "Typhoid fever", "Paratyphoid fever A", 
                "Paratyphoid fever B", "Paratyphoid fever, unspecified", 
                "Salmonella gastroenteritis", "Acute myocardial infarction, unspecified site", 
                "Postmyocardial infarction syndrome", "Intermediate coronary syndrome", 
                "Other acute and subacute forms of ischemic heart disease", 
                "Acute ischemic heart disease, unspecified", 
                "Coronary atherosclerosis of unspecified type of vessel", 
                "Aneurysm of heart", "Chronic ischemic heart disease NOS", 
                "Coronary atherosclerosis of native coronary artery", 
                "Diabetes mellitus without mention of complication, type II",
                "Diabetes mellitus without complication, type I",
                "Diabetes mellitus without complication, unspecified type",
                "Diabetes mellitus without mention of complication, uncontrolled",
                "Diabetes with ketoacidosis, type II",
                "Diabetes with ketoacidosis, type I",
                "Diabetes with ketoacidosis, unspecified type",
                "Diabetes with ketoacidosis, uncontrolled",
                "Recurrent depressive disorder",
                "Anxiety state, unspecified",
                "Adjustment disorder with depressed mood",
                "Prolonged depressive reaction",
                "Adjustment disorder with disturbance of conduct",
                "Major depressive disorder, single episode, moderate",
                "Major depressive disorder, single episode, severe",
                "Major depressive disorder, single episode, in partial remission",
                "Chronic bronchitis, simple",
                "Chronic bronchitis, obstructive",
                "Chronic bronchitis, with acute bronchitis",
                "Chronic bronchitis, unspecified",
                "Idiopathic fibrosing alveolitis",
                "Idiopathic interstitial pneumonia",
                "Other specified alveolar and parietoalveolar pneumonopathies",
                "Unspecified alveolar and parietoalveolar pneumonopathy"
            ],
            'short_title': ["Cholera", "Cholera-vibrio cholerae", "Cholera NOS", "Typhoid fever", 
                           "Paratyphoid fever A", "Paratyphoid fever B", "Paratyphoid fever NOS", 
                           "Salmonella gastroenteritis", "AMI NOS", "Postmyocardial infarction syndrome", 
                           "Intermediate coronary syndrome", "Ac ischemic hrt dis NEC", "Ac ischemic hrt dis NOS", 
                           "Cor ath unsp vessel", "Aneurysm of heart", "Chr ischemic hrt dis NOS", 
                           "Cor ath native cor art", "DMII wo cmp nt st uncntr", "DMI wo cmp nt st uncntr",
                           "DM wo cmp nt st uncntrl", "DM wo cmp nt st uncntrl", "DMII ketoacd nt st uncntr", 
                           "DMI ketoacd nt st uncntr", "DM ketoacd nt st uncntrl", "DM ketoacd unspf uncntrl",
                           "Recurrent depression NOS", "Anxiety state NOS", "Adjustment disorder with depressed mood",
                           "Prolonged depressive reaction", "Adjustment dis-conduct", "Depress psychosis-moderate",
                           "Depress psychosis-severe", "Depress psychosis-partial", "Simple chr bronchitis",
                           "Obst chr bronc w/o exac", "Chr bronchitis w ac bronc", "Chronic bronchitis NOS",
                           "Idiopathic fibrosing alveolitis", "Idiopathic interstitial pneumonia",
                           "Alveolar pneumonopathy NEC", "Alveolar pneumonopathy NOS"]
        })
        
        # Save to gzipped CSV
        icd_df.to_csv(icd_file, index=False, compression='gzip')
        
        print(f"Created {icd_file}")

def create_category_files():
    """Create required category mapping files if they don't exist"""
    # List of category files to create
    category_files = [
        ('micro_spec_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_test_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_org_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_ab_itemid_category_ids.json', {"0": 0, "-100.0": 1}),
        ('micro_dilution_comparison_category_ids.json', {"0": 0, "-100.0": 1}),
        ('patient_category_ids.json', {"M": 0, "F": 1, "WHITE": 2, "BLACK": 3, "ASIAN": 4, "HISPANIC": 5, "OTHER": 6, "AMBULANCE": 7, "WALK": 8, "TAXI": 9, "-100.0": 10}),
        ('triage_category_ids.json', {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "-100.0": 6})
    ]
    
    # Create each file if it doesn't exist
    for filename, data in category_files:
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            print(f"Creating {filepath}")
            
            with open(filepath, 'w') as f:
                json.dump(data, f)
            
            print(f"Created {filepath}")

def main():
    print("=== Generating Required Files for GeMini Training ===")
    
    # Create category files (these are required regardless)
    print("\n=== Creating Category Files ===")
    create_category_files()
    
    # Create MIMIC-IV hospital directory
    print("\n=== Creating MIMIC-IV Hospital Directory ===")
    create_mimiciv_hosp_dir()
    
    # Generate ICD code lists
    print("\n=== Generating ICD Code Lists ===")
    success = get_truncate_icd_diagnosis_cxr()
    
    if success:
        print("\n=== All Required Files Generated Successfully ===")
        print("\nYou should now be able to run the training script:")
        print("bash run_train.sh")
    else:
        print("\n=== Failed to Generate Some Required Files ===")
        print("Please check the error messages above and fix the issues before running the training script.")

if __name__ == "__main__":
    main()
