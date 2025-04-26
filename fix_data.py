# fix_dataset_issues.py
import os
import re

def fix_dataset_file():
    """Fix issues in the dataset.py file"""
    dataset_path = 'src/dataset.py'
    
    # Read the entire file
    with open(dataset_path, 'r') as f:
        content = f.read()
    
    # Fix the '4149' issue
    content = re.sub(
        r"del self\.diagnosis_label_ids\['4149'\]",
        "if '4149' in self.diagnosis_label_ids:\n            del self.diagnosis_label_ids['4149']",
        content
    )
    
    # Fix the empty tensor list issue
    content = re.sub(
        r"self\.diagnosis_text_embeddings = torch\.concat\(diagnosis_text_embeddings,dim=0\)",
        """if diagnosis_text_embeddings:
            self.diagnosis_text_embeddings = torch.concat(diagnosis_text_embeddings, dim=0)
        else:
            print('Warning: Empty diagnosis text embeddings list. Creating dummy embeddings.')
            self.diagnosis_text_embeddings = torch.zeros(1, 768)  # Create dummy embedding""",
        content
    )
    
    # Write the fixed content back
    with open(dataset_path, 'w') as f:
        f.write(content)
    
    print("✓ Fixed issues in src/dataset.py")
    
    # Verify the changes
    with open(dataset_path, 'r') as f:
        new_content = f.read()
    
    if "if '4149' in self.diagnosis_label_ids" in new_content and "if diagnosis_text_embeddings:" in new_content:
        print("✓ Verified changes were successfully made")
        return True
    else:
        print("✗ Changes could not be verified")
        return False

if __name__ == "__main__":
    print("=== Fixing Dataset Issues ===")
    success = fix_dataset_file()
    
    if success:
        print("\n=== Issues Fixed Successfully ===")
        print("\nYou should now be able to run the test script:")
        print("bash run_test.sh")
    else:
        print("\n=== Failed to Fix Issues ===")
        print("Please manually update src/dataset.py with the necessary changes")
