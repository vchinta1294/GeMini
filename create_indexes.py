import os
import json
import pandas as pd
from tqdm import tqdm
import gzip

# Import configuration
from config import MIMIC_CXR_DIR, DATA_DIR, SPLIT_FILE, METADATA_FILE, CHEXPERT_FILE

def create_data_identifiers_from_metadata():
    """
    Create data identifiers from MIMIC-CXR-JPG split and metadata files
    """
    print(f"Loading split file: {SPLIT_FILE}")
    
    # Load the split file which contains subject_id, study_id, and split information
    split_df = pd.read_csv(SPLIT_FILE, compression='gzip')
    print(f"Split file loaded: {len(split_df)} entries")
    
    # Load metadata for additional information
    print(f"Loading metadata file: {METADATA_FILE}")
    metadata_df = pd.read_csv(METADATA_FILE, compression='gzip')
    print(f"Metadata file loaded: {len(metadata_df)} entries")
    
    # Load CheXpert labels
    print(f"Loading CheXpert file: {CHEXPERT_FILE}")
    chexpert_df = pd.read_csv(CHEXPERT_FILE, compression='gzip')
    print(f"CheXpert file loaded: {len(chexpert_df)} entries")
    
    # Merge split and chexpert data to get studies with labels
    study_df = pd.merge(
        split_df[['subject_id', 'study_id', 'split']], 
        chexpert_df[['subject_id', 'study_id']], 
        on=['subject_id', 'study_id'], 
        how='inner'
    )
    print(f"Found {len(study_df)} studies with CheXpert labels")
    
    # Create data identifiers dictionary
    # For this dataset, we'll use a simple mapping:
    # - subject_id: Patient ID
    # - hadm_id: We'll use study_id as a proxy since it's unique per admission
    # - stay_id: We'll use the same study_id since we don't have explicit stay IDs
    
    data_identifiers = {}
    
    for idx, row in tqdm(study_df.iterrows(), total=len(study_df)):
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        
        # In the original code, hadm_id and stay_id might refer to MIMIC-IV admission info
        # Here we'll use study_id as a proxy for both
        hadm_id = study_id
        stay_id = study_id
        
        # Store in the format expected by the preparation script
        data_identifiers[str(idx)] = [subject_id, hadm_id, stay_id]
    
    print(f"Created {len(data_identifiers)} data identifiers")
    return data_identifiers, study_df

def create_dataset_splits(study_df):
    """
    Create train/val/test splits from the dataset split information
    """
    # The split information is already in the split file, so we'll use that
    train_df = study_df[study_df['split'] == 'train']
    val_df = study_df[study_df['split'] == 'validate']
    test_df = study_df[study_df['split'] == 'test']
    
    # Create dictionaries for each split
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    for idx, row in train_df.iterrows():
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        train_dict[str(idx)] = [subject_id, study_id, study_id]
    
    for idx, row in val_df.iterrows():
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        val_dict[str(idx)] = [subject_id, study_id, study_id]
    
    for idx, row in test_df.iterrows():
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        test_dict[str(idx)] = [subject_id, study_id, study_id]
    
    print(f"Created splits: Train ({len(train_dict)}), Val ({len(val_dict)}), Test ({len(test_dict)})")
    return train_dict, val_dict, test_dict

def collect_image_paths():
    """
    Collect all unique image paths from the dataset
    """
    print("Collecting image paths from metadata...")
    metadata_df = pd.read_csv(METADATA_FILE, compression='gzip')
    
    # Print column names to debug
    print("Metadata columns:", metadata_df.columns.tolist())
    
    split_df = pd.read_csv(SPLIT_FILE, compression='gzip')
    print("Split columns:", split_df.columns.tolist())
    
    # Now merge with appropriate column names
    # The split file likely has subject_id and study_id
    image_paths = []
    
    # Use dicom_id from metadata to look up in split file
    for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']
        
        # Pattern is: p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
        prefix = f"p{str(subject_id)[:2]}"
        path = f"{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        image_paths.append(path)
    
    print(f"Collected {len(image_paths)} image paths")
    return image_paths

def main():
    print("Generating dataset indexes from MIMIC-CXR-JPG metadata...")
    
    # Create data identifiers and get study dataframe
    data_identifiers, study_df = create_data_identifiers_from_metadata()
    
    # Save data identifiers
    with open('data/data_identifiers.json', 'w') as f:
        json.dump(data_identifiers, f)
    print(f"Saved data identifiers to data/data_identifiers.json")
    
    # Create dataset splits
    train_dict, val_dict, test_dict = create_dataset_splits(study_df)
    
    # Save splits
    with open('data/train_idx.json', 'w') as f:
        json.dump(train_dict, f)
    print(f"Saved train indexes to data/train_idx.json")
    
    with open('data/val_idx.json', 'w') as f:
        json.dump(val_dict, f)
    print(f"Saved validation indexes to data/val_idx.json")
    
    with open('data/test_idx.json', 'w') as f:
        json.dump(test_dict, f)
    print(f"Saved test indexes to data/test_idx.json")
    
    # Collect image paths
    image_paths = collect_image_paths()
    
    # Save image paths
    with open('unique_image_paths.json', 'w') as f:
        json.dump(image_paths, f)
    print(f"Saved {len(image_paths)} image paths to unique_image_paths.json")
    
    print("Index generation complete!")

if __name__ == "__main__":
    main()
