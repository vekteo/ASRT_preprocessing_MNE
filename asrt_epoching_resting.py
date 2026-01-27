# Load MNE python
import mne
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import os
import os.path as op # Import for robust path joining

### SETUP AND PATHS

# Define constants
DAY = '3'
RESTING_STATE_LABEL = 'resting_start' # resting_start or resting_end
# Annotation marking the start of the resting period (S 83 for before experiment, S 87 for after experiment)
if RESTING_STATE_LABEL == 'resting_start':
    RESTING_ANNOTATION = 'Stimulus/S 83'
else:
    RESTING_ANNOTATION = 'Stimulus/S 87'

BASEPATH = '/Users/teodoravekony/BML-MEMO LAB Dropbox/Lab_workspace/Experiments_Projects/Recent_projects/clinical_brain_projects/TMS_rewiring/Preprocessed_data/EEG/'
EXCLUSIONS = [] # ["162_B", "170_B", "93_L"]
EPOCH_DURATION = 1.0 # Duration of fixed-length epochs (1.0 second)

# Define folders and paths
INPUT_FOLDER = op.join(BASEPATH, 'preproc', 'Day' + DAY)
OUTPUT_FOLDER_EPO = op.join(BASEPATH, 'epoched', 'Day' + DAY, RESTING_STATE_LABEL)

# Define the full path for the CSV file
OUTPUT_CSV_PATH = op.join(BASEPATH, f'epoch_summaries/Day{DAY}_{RESTING_STATE_LABEL}_epoch_summary.csv')
# Define the directory where the CSV will be saved (The parent folder of the CSV file)
OUTPUT_CSV_DIR = op.dirname(OUTPUT_CSV_PATH)

# Create output directories if they don't exist
os.makedirs(OUTPUT_FOLDER_EPO, exist_ok=True)
# Ensure the CSV summary directory exists
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# INITIALIZE LIST TO TRACK EPOCH COUNTS
epoch_summary_list = []

### MAIN LOOP

# Loop through all files in the input folder
for file_name in os.listdir(INPUT_FOLDER):
    
    # 1. Skip excluded files or non-FIF files
    if not file_name.endswith('_preproc.fif'):
        continue
    if any(excl in file_name for excl in EXCLUSIONS):
        print(f"Skipping excluded file: {file_name}")
        continue
        
    # Construct the full file path for reading
    file_path = op.join(INPUT_FOLDER, file_name)
    
    # Extract subject and condition from the file name
    parts = file_name.split('_')
    subject_id = parts[0]
    condition = parts[1] if len(parts) > 1 else 'A' 

    print(f"\n--- Processing file: {file_name} for Subject: {subject_id}, Condition: {condition} ---")

    try:
        # 2. Load the raw data
        raw = mne.io.read_raw_fif(file_path, preload=True)
        
        ##################################### 1. FIND RESTING STATE START TIME #####################################
        
        # Check if the desired annotation exists
        resting_onsets = [ann['onset'] for ann in raw.annotations if ann['description'] == RESTING_ANNOTATION]
        
        if not resting_onsets:
            print(f"Skipping {file_name}: Resting state annotation ({RESTING_ANNOTATION}) not found.")
            continue
            
        # The resting state starts at the first onset of the S 83/ S 87 annotation
        resting_start_time = resting_onsets[0]
        # Define the end time 300 seconds later (5 minutes)
        resting_end_time = resting_start_time + 300

        ##################################### 2. MAKE FIXED-LENGTH EPOCHS #####################################
        
        # FIX for older MNE versions: Crop the raw data first, as 'start' and 'stop' keywords 
        # for make_fixed_length_epochs were introduced in MNE version 1.1.
        raw_cropped = raw.copy().crop(tmin=resting_start_time, tmax=resting_end_time)

        # We use make_fixed_length_epochs directly on the cropped raw data
        epochs = mne.make_fixed_length_epochs(
            raw_cropped,
            duration=EPOCH_DURATION, # 1.0 second epochs
            preload=True, 
            reject_by_annotation=True,
            overlap=0.0
        )
        
        ##################################### 3. REJECT BAD EPOCHS & TRACK STATS #####################################

        reject_criteria = dict(eeg=150e-6) # 150 µV

        # Get the initial number of epochs before rejection
        initial_epochs = len(epochs)
        
        # Apply rejection
        epochs.drop_bad(reject=reject_criteria)

        # Get the number of remaining (good) epochs
        final_epochs = len(epochs)
        rejected_epochs = initial_epochs - final_epochs

        # Append results to the summary list
        epoch_summary_list.append({
            'file_name': file_name,
            'subject_id': subject_id,
            'condition': condition,
            'initial_epochs': initial_epochs,
            'rejected_epochs': rejected_epochs,
            'final_epochs': final_epochs
        })
        
        print(f"Initial epochs: {initial_epochs}, Rejected: {rejected_epochs}, Final: {final_epochs}")
        
        # 4. Save the epoched data
        output_file_path = op.join(OUTPUT_FOLDER_EPO, f'{subject_id}_{condition}_epo.fif')
        epochs.save(output_file_path, overwrite=True)

        print(f"Successfully epoched and saved {subject_id}_{condition} data.")

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        continue # Move to the next file

### SAVE SUMMARY CSV

print("\n--- All files processed. Saving epoch summary CSV. ---")

# Convert the list of dictionaries to a Pandas DataFrame
epoch_summary_df = pd.DataFrame(epoch_summary_list)

# Save the DataFrame to a CSV file
# This now uses the correct full file path, and the directory is guaranteed to exist.
epoch_summary_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Epoch summary saved to: {OUTPUT_CSV_PATH}")
