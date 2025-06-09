import optuna
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


def objective(trial):
    # Sample hyperparameters
    threshold = trial.suggest_float("threshold_voltage", 0.5, 5.0)
    bias = trial.suggest_float("neurons_bias", 5.0, 20.0)
    tau = trial.suggest_float("membrane_time_constant", 10.0, 50.0)
    alpha = trial.suggest_float("fractional_order", 0.6, 0.99)
    memory = trial.suggest_int("memory_length", 10, 50)

    # Build model
    model = FractionalLIFNetwork(
        hidden_layer_size=19,
        output_layer_size=3,
        input_weights=np.eye(19),
        output_weights=np.zeros((19, 3)),
        fractional_order=alpha,
        memory_length=memory,
        membrane_time_constant=tau,
        neurons_bias=bias,
        threshold_voltage=threshold,
        reset_voltage=0.0
    )

    # Use internal validation split
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

    # Extract spike features from training set
    X_feats = []
    for window in X_subtrain:
        result = model.simulate_eeg_classification(window.T, return_traces=True)
        X_feats.append(result["hidden_spikes"].sum(axis=0))
    X_feats = np.vstack(X_feats)

    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_feats, y_subtrain)
    model.output_weights[:] = clf.coef_.T

    # Evaluate on validation set
    y_preds = []
    for window in X_val:
        pred, _ = model.simulate_eeg_classification(window.T)
        y_preds.append(pred)

    return accuracy_score(y_val, y_preds)


data_root = "/u/tmastin/ds004504/derivatives/"
tsv_path = "/u/tmastin/ds004504/participants.tsv"

X, y = load_and_segment_eeg(data_root, tsv_path)
X = X[:20000, :, :1000]  # (n_windows, 19, 1000)
y = y[:20000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)

print("Best parameters:", study.best_params)

