import pandas as pd
import mne
import numpy as np
import os
from sklearn.model_selection import train_test_split # For subject-wise splitting

# --- Your existing helper functions (with minor improvements) ---
def load_tsv_labels(tsv_path):
    """Loads participant labels from a TSV file."""
    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.strip().str.lower()
    # Ensure participant_id is consistently a string for matching
    df['participant_id'] = df['participant_id'].astype(str)
    
    # Define your group mapping (ensure these are the correct categories)
    # Example: A=Alzheimer's, C=Control/Healthy, F=Frontotemporal Dementia
    group_mapping = {'A': 0, 'C': 1, 'F': 2} 
    
    return {row['participant_id']: group_mapping[row['group']] 
            for _, row in df.iterrows() if row['group'] in group_mapping}

def load_eeg_data(file_path):
    """Loads a single EEG .set file using MNE."""
    try:
        # Set verbose=False to minimize MNE's console output during bulk loading
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        # Potential place for last-minute universal preprocessing if needed, e.g.,
        # raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
        # raw.notch_filter(freqs=60, verbose=False) # if powerline in US
        return raw
    except Exception as e:
        print(f"Error loading EEG file {file_path}: {e}")
        return None

def segment_eeg(eeg_data, window_size_samples, stride_samples):
    """Segments EEG data into windows."""
    eeg_array = eeg_data.get_data() * 1e6  # Scale to microvolts
    n_channels, n_timepoints = eeg_array.shape

    if n_timepoints < window_size_samples:
        # print(f"Not enough timepoints ({n_timepoints}) for window size ({window_size_samples}) in one of the files.")
        return np.array([]) # Return empty array if not enough data

    # Ensure the loop range is correct and handles the end of the array
    windows = [eeg_array[:, start:start + window_size_samples]
               for start in range(0, n_timepoints - window_size_samples + 1, stride_samples)]
    
    if not windows: # If the list is empty (e.g., stride > n_timepoints - window_size_samples)
        return np.array([])
        
    return np.array(windows) # Shape: (num_windows, n_channels, window_size_samples)

# --- New functions for subject-aware splitting and processing ---

def process_subject_list(subject_ids_in_set, data_root, label_mapping, 
                         window_size_samples, stride_samples):
    """Loads, segments, and prepares EEG data for a given list of subject IDs."""
    all_windows_for_set, all_labels_for_set = [], []
    processed_subjects_count = 0

    for subject_id in subject_ids_in_set:
        subject_eeg_path = os.path.join(data_root, subject_id, "eeg")
        
        current_subject_label = label_mapping.get(subject_id)
        if current_subject_label is None:
            # This should ideally not happen if subject_ids_in_set are pre-filtered
            print(f"Warning: Label not found for subject {subject_id} during processing. Skipping.")
            continue
            
        if not os.path.isdir(subject_eeg_path):
            print(f"Warning: EEG directory not found for subject {subject_id} at {subject_eeg_path}. Skipping.")
            continue

        subject_had_valid_windows = False
        for file_name in os.listdir(subject_eeg_path):
            if file_name.endswith(".set"):
                eeg_file_path = os.path.join(subject_eeg_path, file_name)
                eeg_data = load_eeg_data(eeg_file_path)
                
                if eeg_data:
                    windows = segment_eeg(eeg_data, window_size_samples, stride_samples)
                    if windows.size > 0: # Check if any windows were created
                        all_windows_for_set.append(windows)
                        all_labels_for_set.append(np.full((windows.shape[0],), current_subject_label))
                        subject_had_valid_windows = True
        
        if subject_had_valid_windows:
            processed_subjects_count += 1
        # else:
        #     print(f"Note: No valid windows found for subject {subject_id}.")


    print(f"  Successfully processed data for {processed_subjects_count} out of {len(subject_ids_in_set)} subjects in this set.")
    if not all_windows_for_set:
        # Return empty arrays with correct number of dimensions for X if no data
        # Assuming n_channels can be inferred or is fixed (e.g. 19)
        # For X: (0_windows, window_size_samples, n_channels_if_known_else_0)
        # Placeholder for n_channels if truly no data. If you know n_channels, use it.
        # Let's assume it's 19 based on your initial problem description.
        num_channels = 19 # Or pass as a parameter if it can vary
        return np.empty((0, window_size_samples, num_channels)), np.empty((0,))

    X_set = np.vstack(all_windows_for_set)  # Shape: (total_windows, n_channels, window_size_samples)
    # Transpose to get (total_windows, window_size_samples, n_channels)
    X_set_transposed = np.transpose(X_set, (0, 2, 1))
    y_set = np.concatenate(all_labels_for_set)
    
    return X_set_transposed, y_set

def prepare_eeg_data_with_splits(data_root, tsv_path, 
                                 window_duration_sec=2, sfreq=500, stride_ratio=0.5,
                                 test_size=0.2, val_size=0.15, random_state=42, num_channels=19):
    """
    Main function to load EEG data, perform subject-aware train/val/test splits,
    segment the data, and return it in the correct shape for a 1D CNN.

    Args:
        data_root (str): Path to the root directory containing subject folders.
        tsv_path (str): Path to the TSV file with participant info and labels.
        window_duration_sec (float): Desired window length in seconds.
        sfreq (int): Sampling frequency of the EEG data.
        stride_ratio (float): Ratio of window_size for stride (e.g., 0.5 for 50% overlap).
                               1.0 means non-overlapping.
        test_size (float): Proportion of subjects for the test set.
        val_size (float): Proportion of subjects for the validation set (from the original total).
        random_state (int): Random seed for reproducible splits.
        num_channels (int): Expected number of channels, used for empty array shapes.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
               Returns (None, ...) for all if initial subject loading fails.
    """
    window_size_samples = int(window_duration_sec * sfreq)
    stride_samples = int(window_size_samples * stride_ratio)

    print(f"Settings: Window Size = {window_size_samples} samples ({window_duration_sec}s), Stride = {stride_samples} samples.")

    label_mapping = load_tsv_labels(tsv_path) # {participant_id_str: label_int}
    if not label_mapping:
        print("Error: No labels loaded. Check TSV path and format.")
        return (None,) * 6

    # Get all subject IDs from data_root that also have labels
    subject_ids_with_labels = []
    subject_labels_for_stratify = []
    for potential_subject_id in os.listdir(data_root):
        # Make sure it's a directory (a subject folder)
        if not os.path.isdir(os.path.join(data_root, potential_subject_id)):
            continue
        
        # Check if this subject_id (folder name) is in our label_mapping
        if potential_subject_id in label_mapping:
            subject_ids_with_labels.append(potential_subject_id)
            subject_labels_for_stratify.append(label_mapping[potential_subject_id])
        # else:
        #     print(f"Debug: Subject folder '{potential_subject_id}' found but not in TSV labels or group invalid.")

    if not subject_ids_with_labels:
        print("Error: No subjects found in data_root that match entries in the TSV label file.")
        return (None,) * 6
    
    print(f"Found {len(subject_ids_with_labels)} subjects with labels to process.")

    # --- Perform subject-wise splits ---
    # 1. Split into (train + validation) and test
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        subject_ids_with_labels,
        subject_labels_for_stratify,
        test_size=test_size,
        stratify=subject_labels_for_stratify, # Stratify by subject labels
        random_state=random_state
    )

    # 2. Split (train + validation) into train and validation
    # Adjust val_size because it's now a fraction of the remaining train_val data
    if len(train_val_ids) == 0: # No data left for training/validation
        print("Warning: No subjects left for training/validation after test split.")
        train_ids, val_ids = [], []
    elif val_size == 0: # User explicitly wants no validation set from the remainder
        train_ids = train_val_ids
        val_ids = []
    else:
        # Calculate val_size relative to the size of train_val_ids
        # (original val_size) / (proportion remaining after test_split)
        relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0
        if relative_val_size >= 1.0 or relative_val_size <=0 : # val_size makes no sense or too small
             # If val_size is problematic (e.g. too large, or 1-test_size is 0)
             # or if train_val_ids is too small to split further meaningfully.
            if len(train_val_ids) > 1 and val_size > 0: # If at least 2 subjects, try to make a val set
                print(f"Warning: val_size ({val_size}) might be too large or too small relative to remaining data. Adjusting split.")
                # Default to a small validation set if possible, e.g. 1 subject or 10% if many.
                # This part might need more sophisticated handling for tiny datasets.
                # For now, let's try to split if possible, otherwise all to train.
                val_split_prop = min(0.25, relative_val_size) if relative_val_size > 0 else 0.25 # Heuristic
                if len(train_val_ids) * val_split_prop < 1 and len(train_val_ids) > 1 : val_split_prop = 1 / len(train_val_ids)


                train_ids, val_ids, _, _ = train_test_split(
                    train_val_ids,
                    train_val_labels,
                    test_size=val_split_prop, # Use adjusted or a default small proportion
                    stratify=train_val_labels,
                    random_state=random_state
                )
            else: # Not enough to split, assign all to train
                 train_ids = train_val_ids
                 val_ids = []
        else: # Normal split for train and validation
            train_ids, val_ids, _, _ = train_test_split(
                train_val_ids,
                train_val_labels,
                test_size=relative_val_size,
                stratify=train_val_labels,
                random_state=random_state
            )

    print(f"\nSubject distribution: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test.")

    # --- Process data for each set ---
    empty_x_shape = (0, window_size_samples, num_channels)
    empty_y_shape = (0,)

    print("\nProcessing Training Data...")
    X_train, y_train = (process_subject_list(train_ids, data_root, label_mapping, window_size_samples, stride_samples)
                        if train_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape)))
    print(f"  Train shapes: X_train-{X_train.shape}, y_train-{y_train.shape}")

    print("\nProcessing Validation Data...")
    X_val, y_val = (process_subject_list(val_ids, data_root, label_mapping, window_size_samples, stride_samples)
                    if val_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape)))
    print(f"  Validation shapes: X_val-{X_val.shape}, y_val-{y_val.shape}")
    
    print("\nProcessing Test Data...")
    X_test, y_test = (process_subject_list(test_ids, data_root, label_mapping, window_size_samples, stride_samples)
                      if test_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape)))
    print(f"  Test shapes: X_test-{X_test.shape}, y_test-{y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test
