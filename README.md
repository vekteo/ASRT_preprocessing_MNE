# ASRT EEG preprocessing & epoching pipeline

This repository contains a pipeline for processing EEG data collected during an **Alternating Serial Reaction Time (ASRT)** task using **MNE-Python**. The scripts facilitate the transition from raw BrainVision recordings to cleaned, epoched data enriched with behavioral metadata.

---

## 📂 Project Structure

The workflow is organized into three distinct phases:

### 1. Preprocessing 
The scripts `asrt_preprocessing.py` and `asrt_preprocessing_day3.py` handle the initial data cleaning for individual subjects.
* **Filtering**: Applies a 50 Hz notch filter to remove line noise.
* **Frequency range**: Implements a band-pass filter from 0.5 to 40 Hz.
* **Artifact removal**: Utilizes Independent Component Analysis (ICA) with 20 components to identify and exclude ocular and muscular artifacts.
* **Bad channel management**: Provides an interactive interface for manual bad channel marking followed by spherical spline interpolation.
* **Re-referencing**: Sets the data to an average EEG reference.

### 2. Task-based epoching
The `asrt_epoching_with_responses_new.py` script segments task data and integrates behavioral performance directly into the metadata.
* **Metadata integration**: Automatically tags epochs with Triplet Type (High/Low), Trial Type (Pattern/Random), and Sequence (A/B).
* **Response mapping**: Matches stimuli with subsequent response triggers to determine accuracy (correct, wrong, or slow) and direction (left, up, down, right).
* **Artifact rejection**: Automatically drops any epoch where the peak-to-peak amplitude exceeds 150 µV.
* **Visualization**: Saves average ERP plots for each subject to ensure data quality.

### 3. Resting-state epoching
The `asrt_epoching_resting.py` script manages 5-minute resting-state segments.
* **Trigger-based cropping**: Uses markers S 83 (before experiment) or S 87 (after experiment) to define the resting window.
* **Fixed-length segments**: Divides the 300-second recording into non-overlapping 1.0-second epochs.
* **Rejection**: Applies a 150 µV rejection threshold to maintain high signal-to-noise ratios.

---

## 🛠 Dependencies

* **Python 3.x**
* **MNE-Python**
* **NumPy**
* **Pandas**
* **Matplotlib** (using the `TkAgg` backend)

---

## 🚀 Getting started

1. **Configuration**: Update the `BASEPATH` variable in the scripts to point to your local project directory.
2. **Preprocessing**: Run the preprocessing scripts to generate `.fif` files. The script will save a summary text file containing the number of removed channels and ICA components for each subject.
3. **Epoching**: Execute the task or resting-state scripts. These will output epoched data into organized subfolders and generate a `CSV` summary log of the epoch counts (initial, rejected, and final).

---

## 📊 Summary Outputs

The pipeline produces an **Epoch Summary CSV** for each run, which includes:
* **Subject ID** and **Condition**.
* **Initial Epochs**: Total segments extracted.
* **Rejected Epochs**: Number of segments lost to the 150 µV threshold.
* **Final Epochs**: Remaining high-quality segments for analysis.
