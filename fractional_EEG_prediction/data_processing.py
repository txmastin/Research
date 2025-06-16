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
    df['participant_id'] = df['participant_id'].astype(str)
    
    # MODIFICATION: Changed group mapping to match the main script's classes
    # Main script uses: 0=Healthy, 1=Alzheimer's, 2=FTD
    # Your original file used: A=0, C=1, F=2. This maps Healthy to 1.
    # Assuming C=Control/Healthy, A=Alzheimer's. Remapping to be consistent.
    group_mapping = {'C': 0, 'A': 1, 'F': 2}
    
    return {row['participant_id']: group_mapping[row['group']]
            for _, row in df.iterrows() if row['group'] in group_mapping}

def load_eeg_data(file_path):
    """Loads a single EEG .set file using MNE."""
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        return raw
    except Exception as e:
        print(f"Error loading EEG file {file_path}: {e}")
        return None

def segment_eeg(eeg_data, window_size_samples, stride_samples):
    """Segments EEG data into windows."""
    eeg_array = eeg_data.get_data() * 1e6
    n_channels, n_timepoints = eeg_array.shape

    if n_timepoints < window_size_samples:
        return np.array([])

    windows = [eeg_array[:, start:start + window_size_samples]
               for start in range(0, n_timepoints - window_size_samples + 1, stride_samples)]
    
    if not windows:
        return np.array([])
        
    return np.array(windows)

# --- MODIFICATION 1: Update process_subject_list to return patient IDs ---
def process_subject_list(subject_ids_in_set, data_root, label_mapping, 
                         window_size_samples, stride_samples, num_channels=19):
    """Loads, segments, and prepares EEG data for a given list of subject IDs."""
    all_windows_for_set, all_labels_for_set = [], []
    # MODIFICATION: Add a list to store patient IDs for each window
    all_patient_ids_for_set = []
    processed_subjects_count = 0

    for subject_id in subject_ids_in_set:
        subject_eeg_path = os.path.join(data_root, subject_id, "eeg")
        
        current_subject_label = label_mapping.get(subject_id)
        if current_subject_label is None:
            print(f"Warning: Label not found for subject {subject_id}. Skipping.")
            continue
            
        if not os.path.isdir(subject_eeg_path):
            print(f"Warning: EEG directory not found for subject {subject_id}. Skipping.")
            continue

        subject_had_valid_windows = False
        for file_name in os.listdir(subject_eeg_path):
            if file_name.endswith(".set"):
                eeg_file_path = os.path.join(subject_eeg_path, file_name)
                eeg_data = load_eeg_data(eeg_file_path)
                
                if eeg_data:
                    windows = segment_eeg(eeg_data, window_size_samples, stride_samples)
                    if windows.size > 0:
                        all_windows_for_set.append(windows)
                        all_labels_for_set.append(np.full((windows.shape[0],), current_subject_label))
                        # MODIFICATION: Add the patient ID for each window created
                        all_patient_ids_for_set.append(np.array([subject_id] * windows.shape[0]))
                        subject_had_valid_windows = True
        
        if subject_had_valid_windows:
            processed_subjects_count += 1

    print(f"  Successfully processed data for {processed_subjects_count} out of {len(subject_ids_in_set)} subjects in this set.")
    if not all_windows_for_set:
        # MODIFICATION: Return three empty arrays now
        return np.empty((0, window_size_samples, num_channels)), np.empty((0,)), np.empty((0,))

    X_set = np.vstack(all_windows_for_set)
    X_set_transposed = np.transpose(X_set, (0, 2, 1))
    y_set = np.concatenate(all_labels_for_set)
    # MODIFICATION: Concatenate the patient IDs as well
    patient_ids_set = np.concatenate(all_patient_ids_for_set)
    
    # MODIFICATION: Return the new patient_ids_set array
    return X_set_transposed, y_set, patient_ids_set

# --- MODIFICATION 2: Update prepare_eeg_data_with_splits to handle 9 return values ---
def prepare_eeg_data_with_splits(data_root, tsv_path, 
                                 window_duration_sec=2, sfreq=500, stride_ratio=0.5,
                                 test_size=0.2, val_size=0.15, random_state=42, num_channels=19):
    """
    Main function to load EEG data, perform subject-aware train/val/test splits,
    segment the data, and return it with corresponding patient IDs.
    """
    window_size_samples = int(window_duration_sec * sfreq)
    stride_samples = int(window_size_samples * stride_ratio)

    print(f"Settings: Window Size = {window_size_samples} samples ({window_duration_sec}s), Stride = {stride_samples} samples.")

    label_mapping = load_tsv_labels(tsv_path)
    if not label_mapping:
        print("Error: No labels loaded. Check TSV path and format.")
        # MODIFICATION: Return 9 values for consistency
        return (None,) * 9

    subject_ids_with_labels = []
    subject_labels_for_stratify = []
    for potential_subject_id in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, potential_subject_id)):
            continue
        
        if potential_subject_id in label_mapping:
            subject_ids_with_labels.append(potential_subject_id)
            subject_labels_for_stratify.append(label_mapping[potential_subject_id])

    if not subject_ids_with_labels:
        print("Error: No subjects found in data_root that match entries in the TSV label file.")
        # MODIFICATION: Return 9 values for consistency
        return (None,) * 9
    
    print(f"Found {len(subject_ids_with_labels)} subjects with labels to process.")

    # --- Perform subject-wise splits ---
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        subject_ids_with_labels,
        subject_labels_for_stratify,
        test_size=test_size,
        stratify=subject_labels_for_stratify,
        random_state=random_state
    )
    
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0
    if len(train_val_ids) > 1 and 0 < relative_val_size < 1:
        train_ids, val_ids, _, _ = train_test_split(
            train_val_ids,
            train_val_labels,
            test_size=relative_val_size,
            stratify=train_val_labels,
            random_state=random_state
        )
    else:
        train_ids = train_val_ids
        val_ids = []


    print(f"\nSubject distribution: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test.")

    # --- Process data for each set ---
    empty_x_shape = (0, window_size_samples, num_channels)
    empty_y_shape = (0,)
    empty_id_shape = (0,)

    print("\nProcessing Training Data...")
    X_train, y_train, train_patient_ids = (process_subject_list(train_ids, data_root, label_mapping, window_size_samples, stride_samples, num_channels)
                        if train_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape), np.empty(empty_id_shape)))
    print(f"  Train shapes: X-{X_train.shape}, y-{y_train.shape}, ids-{train_patient_ids.shape}")

    print("\nProcessing Validation Data...")
    X_val, y_val, val_patient_ids = (process_subject_list(val_ids, data_root, label_mapping, window_size_samples, stride_samples, num_channels)
                    if val_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape), np.empty(empty_id_shape)))
    print(f"  Validation shapes: X-{X_val.shape}, y-{y_val.shape}, ids-{val_patient_ids.shape}")
    
    print("\nProcessing Test Data...")
    X_test, y_test, test_patient_ids = (process_subject_list(test_ids, data_root, label_mapping, window_size_samples, stride_samples, num_channels)
                      if test_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape), np.empty(empty_id_shape)))
    print(f"  Test shapes: X-{X_test.shape}, y-{y_test.shape}, ids-{test_patient_ids.shape}")
    
    # MODIFICATION: Return all 9 arrays
    return X_train, y_train, train_patient_ids, X_val, y_val, val_patient_ids, X_test, y_test, test_patient_ids
