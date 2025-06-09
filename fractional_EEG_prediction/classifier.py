import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from flifnetwork import FractionalLIFNetwork  

def load_tsv_labels(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.strip().str.lower()
    group_mapping = {'A': 0, 'C': 1, 'F': 2}
    return {row['participant_id']: group_mapping[row['group']] for _, row in df.iterrows() if row['group'] in group_mapping}

def load_eeg_data(file_path):
    return mne.io.read_raw_eeglab(file_path, preload=True)

def segment_eeg(eeg_data, window_size, stride):
    eeg_array = eeg_data.get_data() * 1e6
    n_channels, n_timepoints = eeg_array.shape
    windows = [eeg_array[:, start:start + window_size] for start in range(0, n_timepoints - window_size, stride)]
    return np.array(windows)

def load_and_segment_eeg(data_root, tsv_path, window_size=500, stride=500):
    label_mapping = load_tsv_labels(tsv_path)
    all_windows, all_labels = [], []
    for subject in os.listdir(data_root):
        subject_path = os.path.join(data_root, subject, "eeg")
        if os.path.isdir(subject_path):
            for file_name in os.listdir(subject_path):
                if file_name.endswith(".set"):
                    eeg_data = load_eeg_data(os.path.join(subject_path, file_name))
                    windows = segment_eeg(eeg_data, window_size, stride)
                    label = label_mapping.get(subject)
                    if label is not None:
                        all_windows.append(windows)
                        all_labels.append(np.full((windows.shape[0],), label))
    return np.vstack(all_windows), np.concatenate(all_labels)


def evaluate_flif_model(model, X_test, y_test):
    y_pred = []
    for window in X_test:
        window = window.T  # Transpose to shape (time, 19)
        pred, _ = model.simulate_eeg_classification(window)
        y_pred.append(pred)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")
    return acc


data_root = "/u/tmastin/ds004504/derivatives/"
tsv_path = "/u/tmastin/ds004504/participants.tsv"

X, y = load_and_segment_eeg(data_root, tsv_path)
X = X[:, :, :1000]  # (n_windows, 19, 1000)
y = y[:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FLIF parameters
n_inputs = 19
n_outputs = 3
identity_weights = np.eye(n_inputs, dtype=np.float64)
random_output_weights = np.random.rand(n_inputs, n_outputs)

model = FractionalLIFNetwork(
    hidden_layer_size=n_inputs,
    output_layer_size=n_outputs,
    input_weights=identity_weights,
    output_weights=random_output_weights,
    fractional_order=0.733,
    memory_length=200,
    membrane_time_constant=38.0,
    neurons_bias=10.0,
    threshold_voltage=4.777,
    reset_voltage=0.0
)

# Extract spike-based features from training set
X_train_features = []
for window in X_train:
    window = window.T  # Shape (time, channels)
    result = model.simulate_eeg_classification(window, return_traces=True)
    spike_vector = result["hidden_spikes"].sum(axis=0)  # Shape: (19,)
    X_train_features.append(spike_vector)

X_train_features = np.vstack(X_train_features)

# Train a logistic regression classifier
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train_features, y_train)

# Copy trained weights into the FLIF network
model.output_weights[:] = clf.coef_.T  # Shape: (19, 3)

print("Evaluating FLIF-based EEG classifier...")
evaluate_flif_model(model, X_test, y_test)

