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

def build_1d_cnn_eeg_model(input_shape, num_classes=3, learning_rate=0.0001, l2_rate=0.3):
    """
    Builds a 1D Convolutional Neural Network model for EEG classification.

    Args:
        input_shape (tuple): Shape of the input data (window_length_samples, num_channels).
                             Example: (1000, 19) for 1000 samples, 19 channels.
        num_classes (int): Number of output classes (e.g., 3 for Healthy, FTD, Alzheimer's).
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tensorflow.keras.models.Model: The compiled Keras model.
    """
    model = Sequential(name="EEG_1D_CNN")

    # Input Layer - Keras infers this from the first layer's input_shape
    # model.add(Input(shape=input_shape)) # Explicit Input layer can also be used

    # --- Convolutional Block 1 ---
    model.add(Conv1D(filters=16, kernel_size=10, activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1_1', kernel_regularizer=l2(l2_rate)))
    # kernel_size: Length of the 1D convolution window. 
    # A kernel_size of 10 with 500Hz data covers 10/500 = 20ms.
    # You might want to experiment with larger kernel sizes (e.g., 25, 50) 
    # to capture patterns over slightly longer durations (e.g., 50ms, 100ms).
    model.add(BatchNormalization(name='bn1_1'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_1')) 
    # pool_size=2 will halve the sequence length.
    model.add(Dropout(0.4, name='drop1_1'))

    # --- Convolutional Block 2 ---
    model.add(Conv1D(filters=32, kernel_size=10, activation='relu', padding='same', name='conv1_2', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_2'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_2'))
    model.add(Dropout(0.4, name='drop1_2'))

    # --- Convolutional Block 3 ---
    # You can add more blocks like this if needed, increasing filters.
    # For deeper models, ensure your sequence length after pooling is still sufficient.
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu', padding='same', name='conv1_3', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_3'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_3'))
    model.add(Dropout(0.4, name='drop1_3'))

    # --- Feature Aggregation ---
    # GlobalAveragePooling1D calculates the average of features across the time dimension.
    # This makes the model somewhat robust to variations in input length if padding was 'causal'.
    # With 'same' padding and max pooling, the length is reduced, but GAP1D still works well.
    model.add(GlobalAveragePooling1D(name='global_avg_pool'))

    # --- Dense Classification Head ---
    model.add(Dense(units=128, activation='relu', name='dense_1', kernel_regularizer=l2(l2_rate))) # Increased units for more capacity
    model.add(Dropout(0.5, name='drop_dense_1')) # Higher dropout before the final layer

    # --- Output Layer ---
    model.add(Dense(units=num_classes, activation='softmax', name='output_softmax'))
    # units=num_classes: For your three classes.
    # activation='softmax': For multi-class probability output.

    # --- Compile the Model ---
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy', # Use this if y_train are integers (0, 1, 2)
                  # loss='categorical_crossentropy', # Use this if y_train are one-hot encoded
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using the GPU")
    else:
        print("TensorFlow is NOT using the GPU")

    # Example Usage:
    # Define your input shape based on your data
    # (e.g., from X_train.shape[1:])
    WINDOW_LENGTH_SAMPLES = 5000  # Example: 2 seconds * 500 Hz
    NUM_CHANNELS = 19             # Example: 19 EEG channels
    input_shape_for_model = (WINDOW_LENGTH_SAMPLES, NUM_CHANNELS)
    
    NUM_CLASSES = 3 # Healthy, FTD, Alzheimer's

    # Define your paths and parameters
    DATA_ROOT = "/u/tmastin/ds004504/derivatives/"
    TSV_PATH = "/u/tmastin/ds004504/participants.tsv"

    SAMPLING_FREQ = 500  # Hz
    WINDOW_DURATION_SECONDS = 10  # e.g., 2 seconds per window
    STRIDE_OVERLAP_RATIO = 0.2  # 0.5 for 50% overlap, 1.0 for non-overlapping

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


    # ==============================================================================
    #                      STEP 4: ADDRESSING CLASS IMBALANCE
    # ==============================================================================
    from sklearn.utils import class_weight
    import numpy as np

    # This computes weights that give more importance to under-represented classes
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    # Keras expects a dictionary mapping class indices to their weights
    class_weights_dict = dict(enumerate(class_weights))

    print("\n--- Calculated Class Weights ---")
    print(f"Weights for classes 0, 1, 2: {class_weights_dict}")

    # Build the model
    cnn_model = build_1d_cnn_eeg_model(input_shape=input_shape_for_model, num_classes=NUM_CLASSES)

    # Print model summary
    cnn_model.summary()
    
    # --- Define Callbacks ---
    early_stopping_cb = EarlyStopping(monitor='val_loss',  # Monitor validation loss
                                      patience=5,         # Number of epochs with no improvement after which training will be stopped
                                      verbose=1,           # Print message when stopping
                                      restore_best_weights=True) # Restores model weights from the epoch with the best value of the monitored quantity.

    # Define a path to save your best model
    model_checkpoint_cb = ModelCheckpoint(filepath='./best_1d_cnn_model.keras', # .keras is the recommended modern format
                                          save_best_only=True, # Only save a model if `val_loss` has improved
                                          monitor='val_loss',
                                          verbose=1)

    history = cnn_model.fit(X_train, y_train, 
                             epochs=50, 
                             batch_size=64, 
                             validation_data=(X_val, y_val),
                             callbacks=[early_stopping_cb, model_checkpoint_cb],
                             class_weight=class_weights_dict
                            )
    
    # --- And then evaluate it ---
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")





# ==============================================================================
#                      STEP 2: DETAILED MODEL EVALUATION
# ==============================================================================
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Let's define the names of our classes for clear labeling in our plots
CLASS_NAMES = ['Healthy', 'Alzheimer\'s', 'FTD'] # IMPORTANT: Make sure this order is correct!
                                                 # It should correspond to the integer labels (0, 1, 2).

# --- Get Model Predictions ---
# We use .predict() to get the raw softmax probabilities for each class
y_pred_probs = cnn_model.predict(X_test)

# We convert these probabilities to class predictions by taking the argmax
# (i.e., the index of the highest probability)
y_pred = np.argmax(y_pred_probs, axis=1)

# --- Generate and Print Classification Report ---
# This report gives us precision, recall (sensitivity), and F1-score for each class.
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# --- Generate and Visualize Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved to confusion_matrix.png")

# ==============================================================================
#           STEP 3: VISUALIZING LEARNING CURVES
# ==============================================================================

# The 'history' object is returned from the model.fit() call.
history_dict = history.history

# Get the accuracy and loss values
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']

# Generate a sequence of numbers for the x-axis (epochs)
epochs = range(1, len(train_acc) + 1)

# --- Plot Training and Validation Accuracy ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) # 1 row, 2 columns, plot 1
plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# --- Plot Training and Validation Loss ---
plt.subplot(1, 2, 2) # 1 row, 2 columns, plot 2
plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

plt.savefig('learning_curves.png')
print("Learning curves plot saved to learning_curves.png")
