import os

# Base paths for your dataset
MIMIC_CXR_DIR = '/home/vchinta/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0'
DATA_DIR = 'data/MMCaD'

# Make sure these metadata files exist
METADATA_FILE = os.path.join(MIMIC_CXR_DIR, 'mimic-cxr-2.0.0-metadata.csv.gz')
SPLIT_FILE = os.path.join(MIMIC_CXR_DIR, 'mimic-cxr-2.0.0-split.csv.gz')
CHEXPERT_FILE = os.path.join(MIMIC_CXR_DIR, 'mimic-cxr-2.0.0-chexpert.csv.gz')
NEGBIO_FILE = os.path.join(MIMIC_CXR_DIR, 'mimic-cxr-2.0.0-negbio.csv.gz')

# Directories for outputs
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)
os.makedirs('data/test', exist_ok=True)
