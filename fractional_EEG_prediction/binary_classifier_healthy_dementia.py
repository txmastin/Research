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
    Builds a 1D CNN model for BINARY EEG classification (Healthy vs. Dementia).

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

    # --- MODIFICATION: Output Layer for Binary Classification ---
    # A single unit with a sigmoid activation is used for binary classification.
    # It outputs a probability between 0 and 1.
    model.add(Dense(units=1, activation='sigmoid', name='output_sigmoid'))

    # --- MODIFICATION: Compile the Model for Binary Classification ---
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', # Loss function for binary (0/1) classification
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
    
    # This remains the same as it's based on the windowing parameters
    WINDOW_LENGTH_SAMPLES = SAMPLING_FREQ * WINDOW_DURATION_SECONDS
    input_shape_for_model = (WINDOW_LENGTH_SAMPLES, NUM_CHANNELS)

    # --- Load Original 3-Class Data ---
    # The initial labels will be: 0=Healthy, 1=Alzheimer's, 2=FTD
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_eeg_data_with_splits(
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

    if X_train is None or X_train.size == 0:
        print("\nData loading failed or resulted in empty sets. Exiting.")
        exit()

    # ==============================================================================
    #         MODIFICATION: COMBINE ALZHEIMER'S AND FTD CLASSES
    # ==============================================================================
    print("\n--- Remapping labels for binary classification ---")
    # Original: 0=Healthy, 1=Alzheimer's, 2=FTD
    # New:      0=Healthy, 1=Dementia (Combined Alzheimer's and FTD)
    # We achieve this by setting any label that is 2 (FTD) to 1.
    y_train[y_train == 2] = 1
    y_val[y_val == 2] = 1
    y_test[y_test == 2] = 1
    
    # MODIFICATION: Update class names and number of classes for binary task
    NUM_CLASSES = 2
    CLASS_NAMES = ['Healthy', 'Dementia']

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Unique labels in y_train after remapping: {np.unique(y_train)}")

    # ==============================================================================
    #                  STEP 2: HANDLE CLASS IMBALANCE
    # ==============================================================================
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("\n--- Calculated Class Weights for Binary Classification ---")
    print(f"Weights for classes 0 (Healthy) and 1 (Dementia): {class_weights_dict}")

    # ==============================================================================
    #                      STEP 3: BUILD AND TRAIN MODEL
    # ==============================================================================
    cnn_model = build_binary_1d_cnn_eeg_model(input_shape=input_shape_for_model)
    cnn_model.summary()

    # --- Callbacks ---
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model_checkpoint_cb = ModelCheckpoint(filepath='./best_binary_1d_cnn_model.keras', save_best_only=True, monitor='val_loss', verbose=1)

    # --- Train the model ---
    history = cnn_model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping_cb, model_checkpoint_cb],
        class_weight=class_weights_dict
    )

    # ==============================================================================
    #                   STEP 4: EVALUATE THE BINARY MODEL
    # ==============================================================================
    print("\n--- Evaluating model on the test set ---")
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # --- Get Model Predictions ---
    # .predict() on a sigmoid output gives probabilities (shape: [n_samples, 1])
    y_pred_probs = cnn_model.predict(X_test)

    # MODIFICATION: Convert probabilities to binary class predictions (0 or 1)
    # We use a 0.5 threshold to decide the class.
    y_pred = (y_pred_probs > 0.5).astype("int32")

    # --- Generate and Print Classification Report ---
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # --- Generate and Visualize Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix (Healthy vs. Dementia)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('binary_confusion_matrix.png')
    print("Binary confusion matrix saved to binary_confusion_matrix.png")
    plt.close() # Close the figure to prepare for the next plot

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
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # --- Plot Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('binary_learning_curves.png')
    print("Binary learning curves plot saved to binary_learning_curves.png")
    plt.show()
