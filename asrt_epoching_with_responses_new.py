# Load MNE python
import mne
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import os 

### SETUP AND PATHS

# Define constants
DAY = '3'
BASEPATH = '/Users/teodoravekony/BML-MEMO LAB Dropbox/Lab_workspace/Experiments_Projects/Recent_projects/clinical_brain_projects/TMS_rewiring/Preprocessed_data/EEG/'

# Define the folder containing the files to be processed
INPUT_FOLDER = os.path.join(BASEPATH, 'preproc', 'Day' + DAY)
OUTPUT_FOLDER_EPO = os.path.join(BASEPATH, 'epoched', 'Day' + DAY + '/asrt')
OUTPUT_FOLDER_PLOTS = os.path.join(OUTPUT_FOLDER_EPO, 'final_avg_plots')
OUTPUT_CSV_DIR = os.path.join(BASEPATH, f'epoch_summaries') 
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_DIR, f'day{DAY}_epoch_summary.csv')

# Create output directories if they don't exist
os.makedirs(OUTPUT_FOLDER_EPO, exist_ok=True)
os.makedirs(OUTPUT_FOLDER_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# Define all stimulus descriptions for epoching
STIMULUS_DESCRIPTIONS = [
    'Stimulus/S 49', 'Stimulus/S 40', 'Stimulus/S 41', 'Stimulus/S 42', 'Stimulus/S 43', 'Stimulus/S 44',
    'Stimulus/S 45', 'Stimulus/S 46', 'Stimulus/S 47', 'Stimulus/S140', 'Stimulus/S141', 'Stimulus/S142',
    'Stimulus/S143', 'Stimulus/S144', 'Stimulus/S145', 'Stimulus/S146', 'Stimulus/S 10', 'Stimulus/S 11',
    'Stimulus/S 12', 'Stimulus/S 13', 'Stimulus/S 14', 'Stimulus/S 15', 'Stimulus/S 16', 'Stimulus/S110',
    'Stimulus/S111', 'Stimulus/S112', 'Stimulus/S113', 'Stimulus/S114', 'Stimulus/S115', 'Stimulus/S116'
]

# INITIALIZE LIST TO TRACK EPOCH COUNTS
epoch_summary_list = []

### MAIN LOOP

# Loop through all files in the input folder
for file_name in os.listdir(INPUT_FOLDER):
    if file_name.endswith('_preproc.fif'):
        
        # 1. Construct the full file path for reading
        file_path = os.path.join(INPUT_FOLDER, file_name)
        
        # Extract metadata for saving
        parts = file_name.split('_')
        subject_id = parts[0]
        condition = parts[1] if len(parts) > 1 else 'A' 

        print(f"\n--- Processing file: {file_name} ---")
        
        try:
            # 2. Load the raw data and epoch (logic from previous fix remains here)
            data = mne.io.read_raw_fif(file_path, preload=True)

            ##################################### CREATING EPOCHS #####################################
            desired_annotations = mne.Annotations(
                onset=[ann['onset'] for ann in data.annotations if ann['description'] in STIMULUS_DESCRIPTIONS],
                duration=[ann['duration'] for ann in data.annotations if ann['description'] in STIMULUS_DESCRIPTIONS],
                description=[ann['description'] for ann in data.annotations if ann['description'] in STIMULUS_DESCRIPTIONS],
            )

            data.set_annotations(desired_annotations)
            events, event_id = mne.events_from_annotations(data)
            
            if len(events) == 0:
                print(f"Skipping {file_name}: No desired events found.")
                continue

            epochs = mne.Epochs(
                data, events, event_id, tmin=-0.25, tmax=1, baseline=None, preload=True, reject_by_annotation=False
            )
                       
            # --- START Metadata Creation Logic ---
            epochs.metadata = pd.DataFrame({
                "eeg": ["data"] * len(epochs),
                "triplet_type": "NA",
                "trial_type": "NA",
                "rewiring": "NA",
                "sequence": "NA",
                "response": "NA",
                "response_direction": "NA"
            })
            events_code = epochs.events[:, 2] 
            
            def get_ids(descriptions):
                return [event_id.get(desc, -1) for desc in descriptions] 

            H_events = get_ids(['Stimulus/S 10', 'Stimulus/S 11', 'Stimulus/S 14', 'Stimulus/S 15', 'Stimulus/S112', 'Stimulus/S113', 'Stimulus/S114', 'Stimulus/S115'])
            L_events = get_ids(['Stimulus/S 12', 'Stimulus/S 13', 'Stimulus/S 16', 'Stimulus/S110', 'Stimulus/S111', 'Stimulus/S116'])
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, H_events))[0]), "triplet_type"] = "H"
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, L_events))[0]), "triplet_type"] = "L"
            
            P_events = get_ids(['Stimulus/S 11', 'Stimulus/S 13', 'Stimulus/S 15', 'Stimulus/S111', 'Stimulus/S113', 'Stimulus/S114'])
            R_events = get_ids(['Stimulus/S 10', 'Stimulus/S 12', 'Stimulus/S 14', 'Stimulus/S 16', 'Stimulus/S110', 'Stimulus/S112', 'Stimulus/S115', 'Stimulus/S116'])
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, P_events))[0]), "trial_type"] = "P"
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, R_events))[0]), "trial_type"] = "R"

            HH_events = get_ids(['Stimulus/S 14', 'Stimulus/S 15', 'Stimulus/S114', 'Stimulus/S115'])
            HL_events = get_ids(['Stimulus/S 10', 'Stimulus/S 11', 'Stimulus/S110', 'Stimulus/S111'])
            LH_events = get_ids(['Stimulus/S 12', 'Stimulus/S 13', 'Stimulus/S112', 'Stimulus/S113'])
            LL_events = get_ids(['Stimulus/S 16', 'Stimulus/S116'])
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, HH_events))[0]), "rewiring"] = "HH"
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, HL_events))[0]), "rewiring"] = "HL"
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, LH_events))[0]), "rewiring"] = "LH"
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, LL_events))[0]), "rewiring"] = "LL"

            A_events = get_ids(['Stimulus/S 10', 'Stimulus/S 11', 'Stimulus/S 12', 'Stimulus/S 13', 'Stimulus/S 14', 'Stimulus/S 15', 'Stimulus/S 16'])
            B_events = get_ids(['Stimulus/S110', 'Stimulus/S111', 'Stimulus/S112', 'Stimulus/S113', 'Stimulus/S114', 'Stimulus/S115', 'Stimulus/S116'])
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, A_events))[0]), "sequence"] = "A"
            epochs.metadata.loc[epochs.metadata.index.isin(np.where(np.isin(events_code, B_events))[0]), "sequence"] = "B"

            temp_correct = ["NA"] * len(epochs)
            Correct_events = get_ids(['Stimulus/S 40', 'Stimulus/S 41', 'Stimulus/S 42', 'Stimulus/S 43', 'Stimulus/S140', 'Stimulus/S141', 'Stimulus/S142', 'Stimulus/S143'])
            Wrong_events = get_ids(['Stimulus/S 44', 'Stimulus/S 45', 'Stimulus/S 46', 'Stimulus/S 47', 'Stimulus/S144', 'Stimulus/S145', 'Stimulus/S146', 'Stimulus/S147'])
            Slow_events = get_ids(['Stimulus/S 49'])
            
            for i in range(len(epochs.events)):
                event = epochs.events[i, 2]
                if event in Correct_events:
                    temp_correct[i] = "correct"
                elif event in Wrong_events:
                    temp_correct[i] = "wrong"
                elif event in Slow_events:
                    temp_correct[i] = "slow_marker" 
                
            epochs.metadata['temp_correct'] = temp_correct
            
            for i in range(1, len(epochs.metadata)):
                if epochs.metadata.iloc[i-1, epochs.metadata.columns.get_loc("triplet_type")] != "NA":
                    current_correct = epochs.metadata.iloc[i]['temp_correct']
                    
                    if current_correct in ["correct", "wrong"]:
                        epochs.metadata.iloc[i-1, epochs.metadata.columns.get_loc("response")] = current_correct
                    elif current_correct == "slow_marker":
                        epochs.metadata.iloc[i-1, epochs.metadata.columns.get_loc("response")] = "slow"
                    
                    current_event = epochs.events[i, 2]
                    
                    if current_event in get_ids(['Stimulus/S 40', 'Stimulus/S140', 'Stimulus/S 44', 'Stimulus/S144']):
                        epochs.metadata.iloc[i-1, epochs.metadata.columns.get_loc("response_direction")] = "left"
                    elif current_event in get_ids(['Stimulus/S 41', 'Stimulus/S141', 'Stimulus/S 45', 'Stimulus/S145']):
                        epochs.metadata.iloc[i-1, epochs.metadata.columns.get_loc("response_direction")] = "up"
                    elif current_event in get_ids(['Stimulus/S 42', 'Stimulus/S142', 'Stimulus/S 46', 'Stimulus/S146']):
                        epochs.metadata.iloc[i-1, epochs.metadata.columns.get_loc("response_direction")] = "down"
                    elif current_event in get_ids(['Stimulus/S 43', 'Stimulus/S143', 'Stimulus/S 47', 'Stimulus/S147']):
                        epochs.metadata.iloc[i-1, epochs.metadata.columns.get_loc("response_direction")] = "right"

            epochs.metadata = epochs.metadata.drop(columns=["temp_correct"])
            epochs = epochs[epochs.metadata["triplet_type"] != "NA"]

            ##################################### TRACKING EPOCHS #####################################
            
            # Get the initial number of epochs before rejection
            initial_epochs = len(epochs)
            
            ##################################### REJECT BAD EPOCHS #####################################

            reject_criteria = dict(eeg=150e-6)  # 150 µV

            epochs.drop_bad(reject=reject_criteria)
            
            # Get the number of remaining (good) epochs
            final_epochs = len(epochs)
            
            # Calculate the rejected number
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
            
            print(f"Initial: {initial_epochs}, Rejected: {rejected_epochs}, Final: {final_epochs}")
            
            # Plot average of all channels after rejection
            plot_avg = epochs.average().plot(show = False)

            # 7. Save the results
            plot_avg.savefig(os.path.join(OUTPUT_FOLDER_PLOTS, f'{subject_id}_{condition}_avg.png'))
            epochs.save(os.path.join(OUTPUT_FOLDER_EPO, f'{subject_id}_{condition}_epo.fif'), overwrite=True)

            print(f"Successfully epoched and saved {subject_id}_{condition} data.")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue # Move to the next file

### SAVE SUMMARY CSV

print("\n--- All files processed. Saving epoch summary CSV. ---")

# Convert the list of dictionaries to a Pandas DataFrame
epoch_summary_df = pd.DataFrame(epoch_summary_list)

# Save the DataFrame to a CSV file using the correct file path
epoch_summary_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Epoch summary saved to: {OUTPUT_CSV_PATH}")