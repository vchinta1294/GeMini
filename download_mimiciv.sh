#!/bin/bash

# Define variables
USERNAME="vchinta1"  # Replace with your PhysioNet username
MIMIC_IV_DIR="data/mimiciv"

# Create directories
mkdir -p $MIMIC_IV_DIR/hosp

echo "Downloading MIMIC-IV hospital data..."
echo "You will be prompted for your PhysioNet password."

# Download d_labitems.csv.gz (needed for lab test IDs)
wget --user $USERNAME --ask-password -N -c -np \
  https://physionet.org/files/mimiciv/2.2/hosp/d_labitems.csv.gz \
  -P $MIMIC_IV_DIR/hosp/

echo "Download complete!"
