import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Must be set before TensorFlow import

from data_processing import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Dropout, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mode


def build_1d_cnn_eeg_model(input_shape, num_classes=3, learning_rate=0.0001, l2_rate=0.3):
    """
    Builds a 1D Convolutional Neural Network model for EEG classification.
    """
    model = Sequential(name="EEG_1D_CNN")

    # --- Convolutional Block 1 ---
    model.add(Conv1D(filters=16, kernel_size=10, activation='relu', padding='same',
                     input_shape=input_shape, name='conv1_1', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_1'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_1'))
    model.add(Dropout(0.4, name='drop1_1'))

    # --- Convolutional Block 2 ---
    model.add(Conv1D(filters=32, kernel_size=10, activation='relu', padding='same', name='conv1_2', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_2'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_2'))
    model.add(Dropout(0.4, name='drop1_2'))

    # --- Convolutional Block 3 ---
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu', padding='same', name='conv1_3', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_3'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_3'))
    model.add(Dropout(0.4, name='drop1_3'))

    # --- Feature Aggregation ---
    model.add(GlobalAveragePooling1D(name='global_avg_pool'))

    # --- Dense Classification Head ---
    model.add(Dense(units=128, activation='relu', name='dense_1', kernel_regularizer=l2(l2_rate)))
    model.add(Dropout(0.5, name='drop_dense_1'))

    # --- Output Layer ---
    model.add(Dense(units=num_classes, activation='softmax', name='output_softmax'))

    # --- Compile the Model ---
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using the GPU")
    else:
        print("TensorFlow is NOT using the GPU")

    # Define your paths and parameters
    DATA_ROOT = "/u/tmastin/ds004504/derivatives/"
    TSV_PATH = "/u/tmastin/ds004504/participants.tsv"
    SAMPLING_FREQ = 500
    WINDOW_DURATION_SECONDS = 10
    STRIDE_OVERLAP_RATIO = 0.2
    NUM_CHANNELS = 19
    NUM_CLASSES = 3
    CLASS_NAMES = ['Healthy', 'Alzheimer\'s', 'FTD']

    WINDOW_LENGTH_SAMPLES = SAMPLING_FREQ * WINDOW_DURATION_SECONDS
    input_shape_for_model = (WINDOW_LENGTH_SAMPLES, NUM_CHANNELS)

    # ==============================================================================
    #                      MODIFICATION 1: DATA LOADING
    # ==============================================================================
    # NOTE: Your prepare_eeg_data_with_splits function in data_processing.py
    # MUST be updated to return the patient IDs for each window.
    (
        X_train, y_train, train_patient_ids,
        X_val, y_val, val_patient_ids,
        X_test, y_test, test_patient_ids
    ) = prepare_eeg_data_with_splits(
        data_root=DATA_ROOT,
        tsv_path=TSV_PATH,
        window_duration_sec=WINDOW_DURATION_SECONDS,
        sfreq=SAMPLING_FREQ,
        stride_ratio=STRIDE_OVERLAP_RATIO,
        test_size=0.2,
        val_size=0.15,
        random_state=42,
        num_channels=NUM_CHANNELS
    )

    if X_train is not None and X_train.size > 0:
        print(f"\nSuccessfully loaded data.")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"train_patient_ids shape: {train_patient_ids.shape}")
    else:
        print("\nData loading might have encountered issues or resulted in empty sets.")
        exit()

    # ==============================================================================
    #                      CLASS IMBALANCE & MODEL TRAINING
    # ==============================================================================
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("\n--- Calculated Class Weights ---")
    print(f"Weights for classes 0, 1, 2: {class_weights_dict}")

    cnn_model = build_1d_cnn_eeg_model(input_shape=input_shape_for_model, num_classes=NUM_CLASSES)
    cnn_model.summary()

    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint_cb = ModelCheckpoint(filepath='./best_1d_cnn_model.keras', save_best_only=True, monitor='val_loss', verbose=1)

    history = cnn_model.fit(X_train, y_train,
                             epochs=50,
                             batch_size=64,
                             validation_data=(X_val, y_val),
                             callbacks=[early_stopping_cb, model_checkpoint_cb],
                             class_weight=class_weights_dict
                            )

    # ==============================================================================
    #                      WINDOW-LEVEL EVALUATION (Original)
    # ==============================================================================
    print("\n\n" + "="*60)
    print(" " * 15 + "WINDOW-LEVEL EVALUATION")
    print("="*60)

    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"Window-Level Test Loss: {test_loss:.4f}")
    print(f"Window-Level Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = cnn_model.predict(X_test)
    y_pred_window = np.argmax(y_pred_probs, axis=1)

    print("\n--- Window-Level Classification Report ---")
    print(classification_report(y_test, y_pred_window, target_names=CLASS_NAMES))

    print("\n--- Window-Level Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred_window)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Window-Level Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('window_level_confusion_matrix.png')
    plt.close() # Close plot to prevent overlap
    print("Window-level confusion matrix saved to window_level_confusion_matrix.png")


    # ==============================================================================
    #       MODIFICATION 2: PATIENT-LEVEL EVALUATION (MAJORITY VOTING)
    # ==============================================================================
    print("\n\n" + "="*60)
    print(" " * 10 + "PATIENT-LEVEL EVALUATION (MAJORITY VOTING)")
    print("="*60)

    # Create a DataFrame to easily manage the test data results
    results_df = pd.DataFrame({
        'patient_id': test_patient_ids,
        'true_label': y_test,
        'predicted_window_label': y_pred_window
    })

    # Group by patient and aggregate the results
    patient_groups = results_df.groupby('patient_id')

    patient_true_labels = []
    patient_final_predictions = []

    for patient_id, group in patient_groups:
        # The true label is the same for all windows of a patient
        true_label = group['true_label'].iloc[0]
        
        # Perform majority (hard) voting using the mode
        # The mode function returns the most frequent value and its count
        hard_vote_prediction = mode(group['predicted_window_label'])[0]

        patient_true_labels.append(true_label)
        patient_final_predictions.append(hard_vote_prediction)

    # Now calculate and display the patient-level metrics
    num_test_patients = len(patient_groups)
    patient_accuracy = np.sum(np.array(patient_true_labels) == np.array(patient_final_predictions)) / num_test_patients
    
    print(f"\nEvaluation completed for {num_test_patients} unique patients.")
    print(f"Patient-Level Accuracy: {patient_accuracy:.4f}")


    print("\n--- Patient-Level Classification Report ---")
    print(classification_report(patient_true_labels, patient_final_predictions, target_names=CLASS_NAMES))

    print("\n--- Patient-Level Confusion Matrix ---")
    patient_cm = confusion_matrix(patient_true_labels, patient_final_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(patient_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('True Patient-Level Confusion Matrix (Majority Vote)')
    plt.ylabel('Actual Patient Label')
    plt.xlabel('Predicted Patient Label')
    plt.savefig('patient_level_confusion_matrix.png')
    print("Patient-level confusion matrix saved to patient_level_confusion_matrix.png")


    # ==============================================================================
    #           VISUALIZING LEARNING CURVES (Original)
    # ==============================================================================
    history_dict = history.history
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    print("Learning curves plot saved to learning_curves.png")
