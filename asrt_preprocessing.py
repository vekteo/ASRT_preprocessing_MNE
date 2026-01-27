# Load MNE python
import mne
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

# Define subject
subject = '30_E'
day = '1'

# Load data BrainVision eeg data 
raw = mne.io.read_raw_brainvision('C:/Users/teodora.vekony/BML-MEMO LAB Dropbox/Lab_workspace/Experiments_Projects/Recent_projects/clinical_brain_projects/TMS_rewiring/Raw_data/'+subject+'/Day'+day+'/EEG/'+subject+'_Day'+day+'.vhdr', preload=True)
print(raw.info)

##################################### FILTERING #####################################

# Remove noise at 50 Hz
filtered_data = raw.copy()
filtered_data = filtered_data.notch_filter(50) 

# High-pass and low-pass filter the data
filtered_data.filter(0.5, 40)

##################################### REMOVING BAD CHANNELS #####################################

# Plot data
filtered_data.plot(block=True)

# Find bad channels
print(filtered_data.info)

# Remove bad channels
filtered_data.interpolate_bads(reset_bads=False)

##################################### ICA #####################################

# Set up and fit the ICA
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(filtered_data)

# Visualize components
ica.plot_sources(filtered_data, show_scrollbars=False)
ica.plot_components()

# Find and remove the components that are noise
ica.plot_properties(filtered_data, picks=ica.exclude)
ica_data = ica.apply(filtered_data)

##################################### RE-REFERENCING #####################################

# Re-reference the data to the average reference
ica_data.set_eeg_reference('average')

##################################### SAVE PREPROCESSED DATA #####################################

# Save data after filtering, interpolation, ICA, rereferencing
ica_data.save('C:/Users/teodora.vekony/BML-MEMO LAB Dropbox/bml memo members/Teodora_Vekony/GitHub_projects/ASRT_rew_EEG/preproc/Day'+day+'/'+subject+'_Day'+day+'_preproc.fif', overwrite=True)

# Create a new file with first column subject name, second column number of removed channels, third column number of removed components
with open('C:/Users/teodora.vekony/BML-MEMO LAB Dropbox/bml memo members/Teodora_Vekony/GitHub_projects/ASRT_rew_EEG/preproc/Day'+day+'/'+subject+'_Day'+day+'_preproc_info.txt', 'w') as f:
    f.write(subject + '\t' + str((filtered_data.info['bads'])) + '\t' + str(len(ica.exclude)) + '\n')
    f.close()