import numpy as np
import pywt # For Wavelet Decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm # For progress bars

from data_processing import *

# --- 1. Wavelet Feature Extraction ---
def extract_wavelet_features_for_epoch(epoch_data, wavelet_name='db4', level=5):
    """
    Extracts energy features from wavelet decomposition for a single epoch.
    An epoch is expected to have shape (n_timesteps, n_channels).
    """
    n_timesteps, n_channels = epoch_data.shape
    epoch_features = []

    for i in range(n_channels):
        channel_data = epoch_data[:, i]
        coeffs = pywt.wavedec(channel_data, wavelet=wavelet_name, level=level)
        # coeffs is a list: [cA_level, cD_level, cD_level-1, ..., cD1]
        
        # Extract energy from each set of coefficients (approximation and details)
        for c in coeffs:
            energy = np.sum(c**2)
            epoch_features.append(energy)
            # You could add other features here like entropy, std dev of coeffs, etc.
            
    return np.array(epoch_features)

def apply_feature_extraction(X_data, wavelet_name='db4', level=5):
    """
    Applies wavelet feature extraction to all epochs in X_data.
    X_data shape: (n_epochs, n_timesteps, n_channels)
    """
    n_epochs = X_data.shape[0]
    
    # Apply to the first epoch to determine feature dimension
    sample_features = extract_wavelet_features_for_epoch(X_data[0], wavelet_name, level)
    n_features = len(sample_features)
    
    all_features = np.zeros((n_epochs, n_features))
    
    print(f"Extracting wavelet features (wavelet: {wavelet_name}, level: {level})...")
    for i in tqdm(range(n_epochs)):
        all_features[i, :] = extract_wavelet_features_for_epoch(X_data[i], wavelet_name, level)
        
    return all_features

# --- 2. Main Script Logic ---
if __name__ == '__main__':
    # --- Assume your data is loaded here ---
    # Replace these with your actual data loading from the previous scripts.
    # Example shapes based on your previous logs:
    # X_train shape: (43642, 1000, 19)
    # y_train shape: (43642,)
    # X_val shape: (e.g., 5000, 1000, 19)
    # y_val shape: (e.g., 5000,)
    # X_test shape: (e.g., 6000, 1000, 19)
    # y_test shape: (e.g., 6000,)

    # Example: Simulating some data for demonstration if you don't have yours ready
    # These are based on your previous subject distribution, roughly scaled
    # You should replace this with your actual X_train, y_train etc. from `prepare_eeg_data_with_splits`
    
    WINDOW_LENGTH_SAMPLES = 5000  # Example: 2 seconds * 500 Hz
    NUM_CHANNELS = 19             # Example: 19 EEG channels
    input_shape_for_model = (WINDOW_LENGTH_SAMPLES, NUM_CHANNELS)

    NUM_CLASSES = 3 # Healthy, FTD, Alzheimer's

    # Define your paths and parameters
    DATA_ROOT = "/u/tmastin/ds004504/derivatives/"
    TSV_PATH = "/u/tmastin/ds004504/participants.tsv"

    SAMPLING_FREQ = 500  # Hz
    WINDOW_DURATION_SECONDS = 10
    # e.g., 2 seconds per window
    STRIDE_OVERLAP_RATIO = 0.5  # 0.5 for 50% overlap, 1.0 for non-overlapping

    # Call the main preparation function
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_eeg_data_with_splits(
        data_root=DATA_ROOT,
        tsv_path=TSV_PATH,
        window_duration_sec=WINDOW_DURATION_SECONDS,
        sfreq=SAMPLING_FREQ,
        stride_ratio=STRIDE_OVERLAP_RATIO, # 0.5 means 50% overlap
        test_size=0.2,   # 20% of subjects for the test set
        val_size=0.15,   # 15% of (original total) subjects for validation set
        random_state=42, # Ensures splits are the same each time
        num_channels=19  # Your specified number of EEG channels
    )

    # --- You can now use X_train, y_train, X_val, y_val for model training ---

    # Example check:
    if X_train is not None and X_train.size > 0:
        print(f"\nSuccessfully loaded data.")
        print(f"X_train shape: {X_train.shape}") # Should be (num_train_windows, window_samples, 19)
        print(f"y_train shape: {y_train.shape}")
        # Further checks for X_val, X_test as needed
    else:
        print("\nData loading might have encountered issues or resulted in empty sets.")

    # --- Feature Extraction ---
    # Parameters for wavelet decomposition
    WAVELET = 'db4'  # Daubechies 4 - a common choice
    LEVEL = 6        # Number of decomposition levels. 
                     # For 1000 samples at 500Hz, this gets down to ~0-4Hz for Approx coeffs.
                     # This will result in (LEVEL + 1) sets of coefficients per channel.
                     # Total features = NUM_CHANNELS * (LEVEL + 1)

    X_train_features = apply_feature_extraction(X_train, wavelet_name=WAVELET, level=LEVEL)
    X_val_features = apply_feature_extraction(X_val, wavelet_name=WAVELET, level=LEVEL)
    X_test_features = apply_feature_extraction(X_test, wavelet_name=WAVELET, level=LEVEL)

    print(f"Shape of X_train_features: {X_train_features.shape}") # (n_epochs, n_features)

    # --- Feature Scaling ---
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features) # Use transform, not fit_transform
    X_test_scaled = scaler.transform(X_test_features) # Use transform, not fit_transform

    # --- Model Training ---
    print("\nTraining Random Forest model...")
    # You can tune hyperparameters like n_estimators, max_depth, etc.
    # using GridSearchCV or RandomizedSearchCV if needed.
    rf_model = RandomForestClassifier(n_estimators=300, # More trees can be better, up to a point
                                      max_depth=30,      # Limit depth to prevent overfitting
                                      random_state=42,
                                      n_jobs=-1,         # Use all available cores
                                      class_weight='balanced') # Good if you have class imbalance

    rf_model.fit(X_train_scaled, y_train)

    # --- Evaluation ---
    # On Validation Set (optional, but good for hyperparameter tuning if you do it)
    if X_val_scaled.shape[0] > 0:
        print("\nEvaluating on Validation Set...")
        y_val_pred = rf_model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        # print(classification_report(y_val, y_val_pred))

    # On Test Set
    if X_test_scaled.shape[0] > 0:
        print("\nEvaluating on Test Set...")
        y_test_pred = rf_model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print("\nTest Set Classification Report:")
        print(classification_report(y_test, y_test_pred))
    
    print("\nTraditional ML benchmark script finished.")


