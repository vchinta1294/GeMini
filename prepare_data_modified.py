import json
import pickle
from tqdm import tqdm
import pandas as pd
import os
import sys

# Configuration
MIMIC_CXR_DIR = '../mimic/physionet.org/files/mimic-cxr-jpg/2.0.0/'
DATA_DIR = 'data/MMCaD'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('data/train', exist_ok=True)
os.makedirs('data/val', exist_ok=True)
os.makedirs('data/test', exist_ok=True)

# Set CUDA environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"  # Adjust based on your available GPUs

# Check if GPU is available
import torch
if torch.cuda.is_available():
    print(f"GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU available, using CPU")
    sys.exit("GPU required to run this script")

# Function to create dataset indexes if they don't exist
def create_data_indexes():
    # This is a placeholder. You'll need to create these index files 
    # based on your specific dataset needs
    if not os.path.exists('data/train_idx.json'):
        print("Creating train_idx.json (placeholder)")
        with open('data/train_idx.json', 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists('data/val_idx.json'):
        print("Creating val_idx.json (placeholder)")
        with open('data/val_idx.json', 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists('data/test_idx.json'):
        print("Creating test_idx.json (placeholder)")
        with open('data/test_idx.json', 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists('data/data_identifiers.json'):
        print("Creating data_identifiers.json (placeholder)")
        with open('data/data_identifiers.json', 'w') as f:
            json.dump({}, f)

# Load or create data indexes
create_data_indexes()

try:
    with open('data/train_idx.json', 'r') as f:
        train_indexes = json.load(f)

    with open('data/val_idx.json', 'r') as f:
        val_indexes = json.load(f)

    with open('data/test_idx.json', 'r') as f:
        test_indexes = json.load(f)

    with open('data/data_identifiers.json', 'r') as f:
        data_indexes = json.load(f)
    
    print(f"Loaded data indexes: {len(data_indexes)} total samples")
except Exception as e:
    print(f"Error loading indexes: {e}")
    print("Make sure you have created the necessary index files")
    sys.exit(1)

def category2id(category_set):
    category_labels = {}
    category_count = 0.
    for element in category_set:
        category_labels[element] = category_count
        category_count += 1

    return category_labels

def get_parent_icd(code):
    code = str(code)
    try:
        # icd code start with number
        code0 = int(code[0])
        return code[:3]
    except:
        # return code
        if str(code[0]) == 'E':
            return code[:3]
        elif str(code[0]) == 'V':
            return code[:3]
    print('ICD code {0} does not start with number, "E" or "V"'.format(code))
    return code

def get_truncate_icd_diagnosis_cxr():
    with open('data/gem/2015_I10gem.txt', 'r') as f:
        lines = [line.rstrip() for line in f]
        lines = [line.split() for line in lines]
        icd_10 = []
        icd_9 = []
    
        for line in lines:
            icd_10.append(line[0])
            icd_9.append(line[1])
        assert len(icd_10) == len(icd_9)
        icd_gem_10to9 = pd.DataFrame({'icd_10': icd_10, 'icd_9': icd_9})

    all_icd_code_9_list=[]
    indirect_conversions=[]
    for idx, values in tqdm(data_indexes.items()):
        subject_id, hamd_id, stay_id = values
    
        # load 'hosp_ed_cxr_data.csv'
        current_data_path = os.path.join(data_dir, subject_id, hamd_id, stay_id)
        hosp_ed_cxr_data = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
    
        # load diagnosis label
        current_label =pd.read_csv(os.path.join(current_data_path, 'icd_diagnosis.csv'))
    
        # convert icd10 to icd9
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
                    # rarely, there is no conversion from icd10 to icd9, disgard these icd_codes
                    indirect_conversions.append(str(current_label.icd_code.values[i]))
    
        all_icd_code_9_list += icd_9_diagnosis_list

    print('number of indirect_conversions:', len(set(indirect_conversions)))
    print(len(set(all_icd_code_9_list)))

    with open('data/indirect_conversions.json', 'w') as f:
        json.dump(indirect_conversions, f)
    with open('data/all_icd_code9_list.json', 'w') as f:
        json.dump(all_icd_code_9_list, f)

def get_unique_labevent_test_id():
    print("Getting unique lab event test IDs...")
    # Path to the MIMIC-IV hospital data
    d_labevent_path = 'data/mimiciv/hosp/d_labitems.csv.gz'
    
    # Check if the file exists
    if not os.path.exists(d_labevent_path):
        print(f"Warning: {d_labevent_path} does not exist. Creating placeholder.")
        os.makedirs(os.path.dirname(d_labevent_path), exist_ok=True)
        # You would need to download this file from PhysioNet
        print(f"Please download the MIMIC-IV hospital data and place d_labitems.csv.gz in {d_labevent_path}")
        return
    
    d_labevent_dict = pd.read_csv(d_labevent_path, compression='gzip')

    with open('data/unique_labevent_item_id.json', 'w') as f:
        json.dump(list(d_labevent_dict['itemid'].values.astype(str)), f)

    print(f"Saved {len(d_labevent_dict['itemid'])} unique lab event item IDs")
    print(f"Sample item: {d_labevent_dict.loc[0, 'itemid']}")
    print(f"Min item ID: {d_labevent_dict.itemid.min()}")
    sample_item = d_labevent_dict[d_labevent_dict['itemid'] == 50801]
    print(f"Sample item details: {sample_item.to_dict('records') if not sample_item.empty else 'Item not found'}")

def get_unique_category_ids():
    print('Preparing unique category IDs...')
    micro_spec_itemid_category_ids = set()
    micro_test_itemid_category_ids = set()
    micro_org_itemid_category_ids = set()
    micro_ab_itemid_category_ids = set()
    micro_dilution_comparison_category_ids = set()

    patient_category_ids = set()
    triage_category_ids = set()

    # Skip if data_indexes is empty
    if not data_indexes:
        print("Warning: data_indexes is empty. Skipping category ID collection.")
        return

    for idx, values in tqdm(data_indexes.items()):
        subject_id, hamd_id, stay_id = values

        current_data_path = os.path.join(DATA_DIR, subject_id, hamd_id, stay_id)
        
        # Skip if the directory doesn't exist
        if not os.path.exists(current_data_path):
            # print(f"Warning: {current_data_path} does not exist. Skipping.")
            continue

        try:
            hosp_ed_cxr_df = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
        except FileNotFoundError:
            # print(f"Warning: hosp_ed_cxr_data.csv not found in {current_data_path}. Skipping.")
            continue

        try:
            microbiologyevents_df = pd.read_csv(os.path.join(current_data_path, 'microbiologyevents.csv'))
        except FileNotFoundError:
            microbiologyevents_df = pd.DataFrame()
            
        if len(microbiologyevents_df) > 0:
            microbiologyevents_df = microbiologyevents_df.fillna(-100)
            micro_spec_itemid_category_ids = micro_spec_itemid_category_ids | set(
                list(microbiologyevents_df['spec_itemid']))
            micro_test_itemid_category_ids = micro_test_itemid_category_ids | set(
                list(microbiologyevents_df['test_itemid']))
            micro_org_itemid_category_ids = micro_org_itemid_category_ids | set(list(microbiologyevents_df['org_itemid']))
            micro_ab_itemid_category_ids = micro_ab_itemid_category_ids | set(list(microbiologyevents_df['ab_itemid']))
            micro_dilution_comparison_category_ids = micro_dilution_comparison_category_ids | set(
                list(microbiologyevents_df['dilution_comparison']))

        # Handle patient data
        if 'gender' in hosp_ed_cxr_df.columns and 'race' in hosp_ed_cxr_df.columns and 'arrival_transport' in hosp_ed_cxr_df.columns:
            patient_data = hosp_ed_cxr_df.loc[0, ['gender', 'race', 'arrival_transport', 'anchor_age']].fillna(-100.0)
            
            patient_category_ids = patient_category_ids | set([patient_data['gender']])
            patient_category_ids = patient_category_ids | set([patient_data['race']])
            patient_category_ids = patient_category_ids | set([patient_data['arrival_transport']])
        
        # Handle triage data
        triage_columns = ['ed_temperature', 'ed_heartrate', 'ed_resprate', 'ed_o2sat', 'ed_sbp', 'ed_dbp', 'ed_acuity', 'ed_pain']
        if all(col in hosp_ed_cxr_df.columns for col in triage_columns):
            triage_data = hosp_ed_cxr_df.loc[0, triage_columns].fillna(-100.0)
            triage_category_ids = triage_category_ids | set([triage_data['ed_pain']])
            triage_category_ids = triage_category_ids | set([triage_data['ed_acuity']])

    # Convert to category IDs
    micro_spec_itemid_category_ids = category2id(micro_spec_itemid_category_ids)
    micro_test_itemid_category_ids = category2id(micro_test_itemid_category_ids)
    micro_org_itemid_category_ids = category2id(micro_org_itemid_category_ids)
    micro_ab_itemid_category_ids = category2id(micro_ab_itemid_category_ids)
    micro_dilution_comparison_category_ids = category2id(micro_dilution_comparison_category_ids)
    patient_category_ids = category2id(patient_category_ids)
    triage_category_ids = category2id(triage_category_ids)

    # Save to JSON
    print("Saving category IDs...")
    with open('data/micro_spec_itemid_category_ids.json', 'w') as f:
        json.dump(micro_spec_itemid_category_ids, f)
    
    with open('data/micro_test_itemid_category_ids.json', 'w') as f:
        json.dump(micro_test_itemid_category_ids, f)
    
    with open('data/micro_org_itemid_category_ids.json', 'w') as f:
        json.dump(micro_org_itemid_category_ids, f)
    
    with open('data/micro_ab_itemid_category_ids.json', 'w') as f:
        json.dump(micro_ab_itemid_category_ids, f)
    
    with open('data/micro_dilution_comparison_category_ids.json', 'w') as f:
        json.dump(micro_dilution_comparison_category_ids, f)

    with open('data/patient_category_ids.json', 'w') as f:
        json.dump(patient_category_ids, f)

    with open('data/triage_category_ids.json', 'w') as f:
        json.dump(triage_category_ids, f)

    print("Category IDs saved successfully!")

def prepare_discharge_summary_embeddings(data_split_idx_path, split_type='train'):
    print(f"Preparing discharge summary embeddings for {split_type} split...")
    
    # Import here to avoid loading unless function is called
    from transformers import BertTokenizer, AutoModel
    import torch
    
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()

    try:
        with open(data_split_idx_path, 'r') as f:
            data_split_idx = json.load(f)
    except FileNotFoundError:
        print(f"Error: {data_split_idx_path} not found")
        return

    if not data_split_idx:
        print(f"Warning: {data_split_idx_path} is empty")
        return

    discharge_summary_embeddings = {}
    processed_count = 0

    for idx, values in tqdm(data_split_idx.items()):
        subject_id, hamd_id, stay_id = values

        current_data_path = os.path.join(DATA_DIR, subject_id, hamd_id, stay_id)
        
        # Skip if path doesn't exist
        if not os.path.exists(current_data_path):
            continue
            
        try:
            hosp_ed_cxr_df = pd.read_csv(os.path.join(current_data_path, 'hosp_ed_cxr_data.csv'))
        except FileNotFoundError:
            continue
            
        # Check if discharge_note_text column exists
        if 'discharge_note_text' not in hosp_ed_cxr_df.columns:
            continue
            
        discharge_summary = hosp_ed_cxr_df.loc[0, 'discharge_note_text']
        
        # Skip if discharge summary is NaN
        if pd.isna(discharge_summary):
            continue

        with torch.no_grad():
            discharge_summary_tokens = tokenizer(discharge_summary, return_tensors='pt', padding='max_length',
                                                truncation=True, max_length=512)
            discharge_summary_tokens = {key: values.cuda() for key, values in discharge_summary_tokens.items()}
            outputs = model(**discharge_summary_tokens)
            pooler_output = outputs.pooler_output
            discharge_summary_embeddings[idx] = pooler_output.detach().cpu()
            
        processed_count += 1
        
        # Print progress every 100 samples
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} discharge summaries")

    output_path = os.path.join('data', split_type, 'discharge_summary_embeddings.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(discharge_summary_embeddings, f)
        
    print(f"Saved {len(discharge_summary_embeddings)} discharge summary embeddings to {output_path}")

if __name__ == "__main__":
    print("Starting data preparation...")
    
    # Get unique lab event test IDs
    get_unique_labevent_test_id()
    
    # Get unique category IDs
    get_unique_category_ids()
    get_truncate_icd_diagnosis_cxr()
    # Optional: Prepare discharge summary embeddings
    # Uncomment these lines if you want to run them
    # prepare_discharge_summary_embeddings('data/train_idx.json', 'train')
    # prepare_discharge_summary_embeddings('data/val_idx.json', 'val')
    # prepare_discharge_summary_embeddings('data/test_idx.json', 'test')
    
    print("Data preparation complete!")
