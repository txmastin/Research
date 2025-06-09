import pandas as pd
import mne
import numpy as np
import os
from sklearn.model_selection import train_test_split # For subject-wise splitting
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, MaxPooling1D,
    Dropout, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm # For progress bars (used in original feature extraction)

# --- Data Processing Functions (Integrated and Modified) ---
def load_tsv_labels_binary(tsv_path, group_A_label='A', group_C_label='C'):
    """
    Loads participant labels from a TSV file for binary classification
    between specified group_A_label (new class 0) and group_C_label (new class 1).
    """
    print(f"Loading labels for binary classification: '{group_A_label}' vs '{group_C_label}'")
    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.strip().str.lower()
    df['participant_id'] = df['participant_id'].astype(str)
    
    # Define the groups to keep and their new binary mapping
    groups_to_keep = [group_A_label, group_C_label]
    binary_label_mapping = {group_A_label: 0, group_C_label: 1}
    
    df_filtered = df[df['group'].isin(groups_to_keep)]
    
    return {row['participant_id']: binary_label_mapping[row['group']] 
            for _, row in df_filtered.iterrows() if row['group'] in binary_label_mapping}

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
    eeg_array = eeg_data.get_data() * 1e6  # Scale to microvolts
    n_channels, n_timepoints = eeg_array.shape

    if n_timepoints < window_size_samples:
        return np.array([]) 

    windows = [eeg_array[:, start:start + window_size_samples]
               for start in range(0, n_timepoints - window_size_samples + 1, stride_samples)]
    
    if not windows: 
        return np.array([])
        
    return np.array(windows)

def process_subject_list(subject_ids_in_set, data_root, label_mapping, 
                         window_size_samples, stride_samples, num_channels_expected=19):
    """Loads, segments, and prepares EEG data for a given list of subject IDs."""
    all_windows_for_set, all_labels_for_set = [], []
    processed_subjects_count = 0

    for subject_id in subject_ids_in_set:
        subject_eeg_path = os.path.join(data_root, subject_id, "eeg")
        
        current_subject_label = label_mapping.get(subject_id)
        if current_subject_label is None:
            print(f"Warning: Label not found for subject {subject_id} during processing (should not happen if lists are pre-filtered). Skipping.")
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
                    if eeg_data.info['nchan'] != num_channels_expected:
                        print(f"Warning: Subject {subject_id}, file {file_name} has {eeg_data.info['nchan']} channels, expected {num_channels_expected}. Skipping file.")
                        continue
                    windows = segment_eeg(eeg_data, window_size_samples, stride_samples)
                    if windows.size > 0: 
                        all_windows_for_set.append(windows)
                        all_labels_for_set.append(np.full((windows.shape[0],), current_subject_label))
                        subject_had_valid_windows = True
        
        if subject_had_valid_windows:
            processed_subjects_count += 1

    print(f"  Successfully processed data for {processed_subjects_count} out of {len(subject_ids_in_set)} subjects in this set.")
    if not all_windows_for_set:
        return np.empty((0, window_size_samples, num_channels_expected)), np.empty((0,))

    X_set = np.vstack(all_windows_for_set)
    X_set_transposed = np.transpose(X_set, (0, 2, 1))
    y_set = np.concatenate(all_labels_for_set)
    
    return X_set_transposed, y_set

def prepare_eeg_data_with_splits_binary(data_root, tsv_path, 
                                 group_A_label='A', group_C_label='C', # Specify which original groups are new 0 and 1
                                 window_duration_sec=10, sfreq=500, stride_ratio=0.5,
                                 test_size=0.2, val_size=0.15, random_state=42, num_channels=19):
    window_size_samples = int(window_duration_sec * sfreq)
    stride_samples = int(window_size_samples * stride_ratio)

    print(f"Settings for Binary Classification: Window Size = {window_size_samples} samples ({window_duration_sec}s), Stride = {stride_samples} samples.")

    # Use the modified label loader for binary classification
    label_mapping = load_tsv_labels_binary(tsv_path, group_A_label=group_A_label, group_C_label=group_C_label)
    if not label_mapping:
        print("Error: No labels loaded for binary task. Check TSV path and specified groups.")
        return (None,) * 6

    subject_ids_with_labels = []
    subject_labels_for_stratify = []
    for potential_subject_id in os.listdir(data_root):
        if not os.path.isdir(os.path.join(data_root, potential_subject_id)):
            continue
        if potential_subject_id in label_mapping: # label_mapping now only contains subjects from the two selected groups
            subject_ids_with_labels.append(potential_subject_id)
            subject_labels_for_stratify.append(label_mapping[potential_subject_id])

    if not subject_ids_with_labels:
        print("Error: No subjects found in data_root that match the selected binary classes in the TSV label file.")
        return (None,) * 6
    
    print(f"Found {len(subject_ids_with_labels)} subjects for binary classification task.")
    if len(subject_ids_with_labels) < 2 : # Need at least 2 subjects to attempt a split
        print("Error: Fewer than 2 subjects available for splitting. Cannot proceed.")
        return (None,) * 6


    # --- Perform subject-wise splits ---
    # Ensure enough subjects for all splits
    min_subjects_for_split = 2 # For train_test_split to work without error for stratification
    if len(np.unique(subject_labels_for_stratify)) < 2:
        print("Warning: Only one class present after filtering. Cannot stratify or perform meaningful classification. Check your class selection.")
        # Fallback to non-stratified split if only one class, though this dataset is problematic
        stratify_option_main = None
    else:
        stratify_option_main = subject_labels_for_stratify
    
    num_test_subjects = int(np.ceil(len(subject_ids_with_labels) * test_size))
    if num_test_subjects < min_subjects_for_split and len(subject_ids_with_labels) > num_test_subjects:
         num_test_subjects = min(min_subjects_for_split, len(subject_ids_with_labels) -1 if len(subject_ids_with_labels) > 1 else 0)


    if len(subject_ids_with_labels) - num_test_subjects < min_subjects_for_split : # Not enough for train_val
         print("Warning: Not enough subjects to create a meaningful train/val and test split according to test_size. Adjusting sizes or stopping.")
         # Simplified: if not enough for test, put all in train_val, no test set
         if len(subject_ids_with_labels) <= min_subjects_for_split : # very few subjects
            train_val_ids = subject_ids_with_labels
            train_val_labels = subject_labels_for_stratify
            test_ids = []
            print("  Too few subjects. Using all for train/validation, no test set created.")
         else: # Try to make a minimal test set
            train_val_ids, test_ids, train_val_labels, _ = train_test_split(
                subject_ids_with_labels, subject_labels_for_stratify,
                test_size=num_test_subjects if num_test_subjects > 0 else 1, # Ensure at least 1 for test if possible
                stratify=stratify_option_main, random_state=random_state )
    else:
        train_val_ids, test_ids, train_val_labels, _ = train_test_split(
            subject_ids_with_labels, subject_labels_for_stratify,
            test_size=num_test_subjects, stratify=stratify_option_main, random_state=random_state)


    num_val_subjects = int(np.ceil(len(train_val_ids) * (val_size / (1.0 - test_size if test_size < 1.0 else 1.0))))
    if num_val_subjects < min_subjects_for_split and len(train_val_ids) > num_val_subjects :
        num_val_subjects = min(min_subjects_for_split, len(train_val_ids) -1 if len(train_val_ids) > 1 else 0)

    if len(train_val_ids) == 0:
        train_ids, val_ids = [], []
    elif len(train_val_ids) - num_val_subjects < min_subjects_for_split or num_val_subjects == 0:
        print("Warning: Not enough subjects in train_val to create a meaningful validation split. Using all train_val for training.")
        train_ids = train_val_ids
        val_ids = []
    else:
        if len(np.unique(train_val_labels)) < 2: stratify_option_val = None
        else: stratify_option_val = train_val_labels
        train_ids, val_ids, _, _ = train_test_split(
            train_val_ids, train_val_labels,
            test_size=num_val_subjects, stratify=stratify_option_val, random_state=random_state)
        
    print(f"\nSubject distribution: {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test.")

    empty_x_shape = (0, window_size_samples, num_channels)
    empty_y_shape = (0,)

    print("\nProcessing Training Data...")
    X_train, y_train = (process_subject_list(train_ids, data_root, label_mapping, window_size_samples, stride_samples, num_channels)
                        if train_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape)))
    print(f"  Train shapes: X_train-{X_train.shape}, y_train-{y_train.shape}")

    print("\nProcessing Validation Data...")
    X_val, y_val = (process_subject_list(val_ids, data_root, label_mapping, window_size_samples, stride_samples, num_channels)
                    if val_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape)))
    print(f"  Validation shapes: X_val-{X_val.shape}, y_val-{y_val.shape}")
    
    print("\nProcessing Test Data...")
    X_test, y_test = (process_subject_list(test_ids, data_root, label_mapping, window_size_samples, stride_samples, num_channels)
                      if test_ids else (np.empty(empty_x_shape), np.empty(empty_y_shape)))
    print(f"  Test shapes: X_test-{X_test.shape}, y_test-{y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# --- Model Building Function (Integrated and Modified) ---
def build_1d_cnn_eeg_binary_model(input_shape, learning_rate=0.0001, l2_rate=0.01):
    """
    Builds a 1D CNN model for BINARY EEG classification.
    """
    model = Sequential(name="EEG_1D_CNN_Binary")

    model.add(Conv1D(filters=8, kernel_size=10, activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1_1', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_1'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_1')) 
    model.add(Dropout(0.25, name='drop1_1'))

    model.add(Conv1D(filters=16, kernel_size=10, activation='relu', padding='same', name='conv1_2', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_2'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_2'))
    model.add(Dropout(0.25, name='drop1_2'))

    model.add(Conv1D(filters=32, kernel_size=10, activation='relu', padding='same', name='conv1_3', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_3'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_3'))
    model.add(Dropout(0.25, name='drop1_3'))

    model.add(GlobalAveragePooling1D(name='global_avg_pool'))

    model.add(Dense(units=64, activation='relu', name='dense_1', kernel_regularizer=l2(l2_rate)))
    model.add(Dropout(0.5, name='drop_dense_1'))

    # --- Output Layer for Binary Classification ---
    model.add(Dense(units=1, activation='sigmoid', name='output_sigmoid'))

    # --- Compile the Model for Binary Classification ---
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# --- Main Execution Block ---
if __name__ == '__main__':
    # Set to "" to use CPU, or remove/comment out to attempt GPU usage
    # os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # if tf.config.list_physical_devices('GPU'):
    #     print("TensorFlow is using the GPU")
    # else:
    #     print("TensorFlow is NOT using the GPU")

    # --- Parameters for Data Loading & Model ---
    # These paths need to be correct for your system
    DATA_ROOT = "/u/tmastin/ds004504/derivatives/" 
    TSV_PATH = "/u/tmastin/ds004504/participants.tsv"

    # Specify which original groups to keep for binary classification
    # Assuming original 'A' maps to 0, 'C' maps to 1, 'F' maps to 2
    # We will keep 'A' (new class 0) and 'C' (new class 1)
    GROUP_FOR_CLASS_0 = 'A' 
    GROUP_FOR_CLASS_1 = 'C'

    SAMPLING_FREQ = 500
    WINDOW_DURATION_SECONDS = 10 # Using the 10-second window that showed good results
    STRIDE_OVERLAP_RATIO = 0.5
    NUM_CHANNELS = 19 # Should match your data

    # Data Loading for Binary Task
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_eeg_data_with_splits_binary(
        data_root=DATA_ROOT,
        tsv_path=TSV_PATH,
        group_A_label=GROUP_FOR_CLASS_0, # Will be mapped to 0
        group_C_label=GROUP_FOR_CLASS_1, # Will be mapped to 1
        window_duration_sec=WINDOW_DURATION_SECONDS,
        sfreq=SAMPLING_FREQ,
        stride_ratio=STRIDE_OVERLAP_RATIO,
        test_size=0.2,
        val_size=0.15,
        random_state=42,
        num_channels=NUM_CHANNELS
    )

    if X_train is None or X_train.size == 0:
        print("\nData loading failed or X_train is empty for binary task. Exiting.")
        exit()
    
    print(f"\nSuccessfully loaded data for binary task.")
    print(f"X_train shape: {X_train.shape}, y_train unique labels: {np.unique(y_train, return_counts=True)}")
    if X_val.size > 0:
        print(f"X_val shape: {X_val.shape}, y_val unique labels: {np.unique(y_val, return_counts=True)}")
    if X_test.size > 0:
        print(f"X_test shape: {X_test.shape}, y_test unique labels: {np.unique(y_test, return_counts=True)}")


    # Model Building
    input_shape_for_model = (X_train.shape[1], X_train.shape[2]) # (WINDOW_LENGTH_SAMPLES, NUM_CHANNELS)
    LEARNING_RATE = 0.0001 # From your successful experiments
    L2_REG_RATE = 0.01   # From your successful experiments

    cnn_model = build_1d_cnn_eeg_binary_model(
        input_shape=input_shape_for_model,
        learning_rate=LEARNING_RATE,
        l2_rate=L2_REG_RATE
    )
    cnn_model.summary()

    # Callbacks
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint_cb = ModelCheckpoint(filepath='./best_binary_cnn_model.keras',
                                          save_best_only=True, monitor='val_loss', verbose=1)

    # For a "quick test example", train for fewer epochs
    QUICK_TEST_EPOCHS = 50 

    print(f"\n--- Starting QUICK TEST training for {QUICK_TEST_EPOCHS} epochs ---")
    if X_val.size > 0 and y_val.size > 0 : # Ensure validation data is not empty
        history = cnn_model.fit(X_train, y_train,
                                epochs=QUICK_TEST_EPOCHS, # Reduced for quick test
                                batch_size=64,
                                validation_data=(X_val, y_val),
                                callbacks=[early_stopping_cb, model_checkpoint_cb]
                               )
    elif X_train.size > 0 and y_train.size > 0:
        print("Warning: Validation set is empty. Training without validation.")
        history = cnn_model.fit(X_train, y_train,
                                epochs=QUICK_TEST_EPOCHS,
                                batch_size=64,
                                callbacks=[model_checkpoint_cb] # EarlyStopping needs val_loss
                               )
    else:
        print("Training data is empty. Cannot fit model.")
        history = None # Or handle appropriately


    # Evaluation (on test set, using the best model restored by EarlyStopping or loaded from checkpoint)
    if X_test.size > 0 and y_test.size > 0 and history is not None:
        print("\n--- Evaluating on Test Set (using best model from training) ---")
        # If EarlyStopping had restore_best_weights=True, cnn_model is already the best.
        # Otherwise, load from ModelCheckpoint:
        # print("Loading best model from checkpoint for evaluation...")
        # cnn_model = tf.keras.models.load_model('./best_binary_cnn_model.keras')
        
        test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # You might want to generate predictions and a full classification report for binary too
        # from sklearn.metrics import classification_report
        # y_pred_proba = cnn_model.predict(X_test)
        # y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1)
        # print(classification_report(y_test, y_pred, target_names=[GROUP_FOR_CLASS_0, GROUP_FOR_CLASS_1]))

    elif history is None:
        print("Model was not trained. Skipping evaluation.")
    else:
        print("\nTest set is empty. Skipping evaluation.")
        
    print("\nBinary classification test script finished.")

