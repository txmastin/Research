import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D,
    Dropout, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# Assume data_processing.py exists and contains the necessary function
from data_processing import prepare_eeg_data_with_splits

os.environ["CUDA_VISIBLE_DEVICES"] = "" # Must be set before TensorFlow import

def build_binary_1d_cnn_eeg_model(input_shape, learning_rate=0.0001, l2_rate=0.3):
    """
    Builds a 1D CNN model for BINARY EEG classification.
    This is versatile enough for Healthy vs. Dementia or Alzheimer's vs. FTD.

    Args:
        input_shape (tuple): Shape of the input data (window_length_samples, num_channels).
        learning_rate (float): Learning rate for the optimizer.
        l2_rate (float): L2 regularization factor.

    Returns:
        tensorflow.keras.models.Model: The compiled Keras model.
    """
    model = Sequential(name="EEG_Binary_1D_CNN")

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

    # --- Output Layer for Binary Classification ---
    model.add(Dense(units=1, activation='sigmoid', name='output_sigmoid'))

    # --- Compile the Model for Binary Classification ---
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    # ==============================================================================
    #                      STEP 1: LOAD AND PREPARE DATA
    # ==============================================================================
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using the GPU")
    else:
        print("TensorFlow is NOT using the GPU")

    # --- Data Parameters ---
    DATA_ROOT = "/u/tmastin/ds004504/derivatives/"
    TSV_PATH = "/u/tmastin/ds004504/participants.tsv"
    SAMPLING_FREQ = 500
    WINDOW_DURATION_SECONDS = 10
    STRIDE_OVERLAP_RATIO = 0.2
    NUM_CHANNELS = 19
    WINDOW_LENGTH_SAMPLES = SAMPLING_FREQ * WINDOW_DURATION_SECONDS
    input_shape_for_model = (WINDOW_LENGTH_SAMPLES, NUM_CHANNELS)

    # --- Load Original 3-Class Data ---
    # The initial labels will be: 0=Healthy, 1=Alzheimer's, 2=FTD
    X_train_full, y_train_full, X_val_full, y_val_full, X_test_full, y_test_full = prepare_eeg_data_with_splits(
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

    if X_train_full is None or X_train_full.size == 0:
        print("\nData loading failed or resulted in empty sets. Exiting.")
        exit()

    # ==============================================================================
    #    MODIFICATION: FILTER FOR DEMENTIA CLASSES & REMAP LABELS
    # ==============================================================================
    print("\n--- Filtering data for Alzheimer's vs. FTD classification ---")

    # Define the new classes for this specific task
    CLASS_NAMES = ['Alzheimer\'s', 'FTD']

    # --- Filter and Remap Training Data ---
    train_mask = (y_train_full == 1) | (y_train_full == 2)
    X_train = X_train_full[train_mask]
    y_train = y_train_full[train_mask]
    y_train[y_train == 1] = 0  # Alzheimer's -> 0
    y_train[y_train == 2] = 1  # FTD -> 1

    # --- Filter and Remap Validation Data ---
    val_mask = (y_val_full == 1) | (y_val_full == 2)
    X_val = X_val_full[val_mask]
    y_val = y_val_full[val_mask]
    y_val[y_val == 1] = 0  # Alzheimer's -> 0
    y_val[y_val == 2] = 1  # FTD -> 1

    # --- Filter and Remap Test Data ---
    test_mask = (y_test_full == 1) | (y_test_full == 2)
    X_test = X_test_full[test_mask]
    y_test = y_test_full[test_mask]
    y_test[y_test == 1] = 0  # Alzheimer's -> 0
    y_test[y_test == 2] = 1  # FTD -> 1

    print(f"Original X_train shape: {X_train_full.shape}")
    print(f"Filtered X_train shape for AD vs FTD: {X_train.shape}")
    print(f"Unique labels in new y_train set: {np.unique(y_train)}")

    # ==============================================================================
    #                  STEP 2: HANDLE CLASS IMBALANCE
    # ==============================================================================
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("\n--- Calculated Class Weights for AD vs FTD Classification ---")
    print(f"Weights for classes 0 (Alzheimer's) and 1 (FTD): {class_weights_dict}")

    # ==============================================================================
    #                      STEP 3: BUILD AND TRAIN MODEL
    # ==============================================================================
    # We can reuse the same binary model architecture
    cnn_model = build_binary_1d_cnn_eeg_model(input_shape=input_shape_for_model)
    cnn_model.summary()

    # --- Callbacks (with new filename) ---
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint_cb = ModelCheckpoint(filepath='./best_dementia_binary_model.keras', save_best_only=True, monitor='val_loss', verbose=1)

    # --- Train the model on the filtered data ---
    history = cnn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping_cb, model_checkpoint_cb],
        class_weight=class_weights_dict
    )

    # ==============================================================================
    #             STEP 4: EVALUATE THE ALZHEIMER'S vs FTD MODEL
    # ==============================================================================
    print("\n--- Evaluating model on the AD vs FTD test set ---")
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # --- Get Model Predictions ---
    y_pred_probs = cnn_model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    # --- Generate and Print Classification Report ---
    print("\n--- AD vs FTD Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # --- Generate and Visualize Confusion Matrix (with new filename) ---
    print("\n--- AD vs FTD Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Alzheimer\'s vs. FTD)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('dementia_binary_confusion_matrix.png')
    print("Dementia binary confusion matrix saved to dementia_binary_confusion_matrix.png")
    plt.close()

    # ==============================================================================
    #                   STEP 5: VISUALIZE LEARNING CURVES
    # ==============================================================================
    history_dict = history.history
    train_acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    train_loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(12, 5))

    # --- Plot Accuracy ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('AD vs FTD: Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # --- Plot Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('AD vs FTD: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('dementia_binary_learning_curves.png')
    print("Dementia binary learning curves plot saved to dementia_binary_learning_curves.png")
    plt.show()
