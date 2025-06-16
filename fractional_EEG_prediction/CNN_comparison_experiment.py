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

# UPDATE your build_1d_cnn_eeg_model function
def build_1d_cnn_eeg_model(input_shape, num_classes=3, learning_rate=0.0001, l2_rate=0.3, dropout_rate=0.4, num_blocks=3, initial_filters=16, dense_units=128):
    """
    Builds a flexible 1D CNN model for EEG classification.
    """
    model = Sequential(name=f"EEG_CNN_{num_blocks}blocks_{initial_filters}filters")

    # Input Layer
    model.add(Conv1D(filters=initial_filters, kernel_size=10, activation='relu', padding='same', 
                     input_shape=input_shape, name='conv1_1', kernel_regularizer=l2(l2_rate)))
    model.add(BatchNormalization(name='bn1_1'))
    model.add(MaxPooling1D(pool_size=2, name='pool1_1')) 
    model.add(Dropout(dropout_rate, name='drop1_1'))

    # Add subsequent blocks in a loop
    current_filters = initial_filters
    for i in range(1, num_blocks):
        current_filters *= 2 # Double filters for each block
        model.add(Conv1D(filters=current_filters, kernel_size=10, activation='relu', padding='same', name=f'conv1_{i+1}', kernel_regularizer=l2(l2_rate)))
        model.add(BatchNormalization(name=f'bn1_{i+1}'))
        model.add(MaxPooling1D(pool_size=2, name=f'pool1_{i+1}'))
        model.add(Dropout(dropout_rate, name=f'drop1_{i+1}'))

    # Feature Aggregation and Classification Head
    model.add(GlobalAveragePooling1D(name='global_avg_pool'))
    model.add(Dense(units=dense_units, activation='relu', name='dense_1', kernel_regularizer=l2(l2_rate)))
    model.add(Dropout(0.5, name='drop_dense_1')) # Keep final dropout high
    model.add(Dense(units=num_classes, activation='softmax', name='output_softmax'))

    # Compile
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model



#
# REPLACE EVERYTHING FROM THIS POINT DOWN IN YOUR SCRIPT
#

if __name__ == '__main__':
    # --- Your data loading and parameter setup is fine, it stays the same ---
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if tf.config.list_physical_devices('GPU'):
        print("TensorFlow is using the GPU")
    else:
        print("TensorFlow is NOT using the GPU")

    WINDOW_LENGTH_SAMPLES = 5000
    NUM_CHANNELS = 19
    input_shape_for_model = (WINDOW_LENGTH_SAMPLES, NUM_CHANNELS)
    NUM_CLASSES = 3
    DATA_ROOT = "/u/tmastin/ds004504/derivatives/"
    TSV_PATH = "/u/tmastin/ds004504/participants.tsv"
    SAMPLING_FREQ = 500
    WINDOW_DURATION_SECONDS = 10
    STRIDE_OVERLAP_RATIO = 0.2

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_eeg_data_with_splits(
        data_root=DATA_ROOT,
        tsv_path=TSV_PATH,
        window_duration_sec=WINDOW_DURATION_SECONDS,
        sfreq=SAMPLING_FREQ,
        stride_ratio=STRIDE_OVERLAP_RATIO,
        test_size=0.2,
        val_size=0.15,
        random_state=42,
        num_channels=19
    )

    if X_train is not None and X_train.size > 0:
        print(f"\nSuccessfully loaded data.")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
    else:
        print("\nData loading might have encountered issues or resulted in empty sets.")

    from sklearn.utils import class_weight
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    # --- Calculate Class Weights ---
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    print("\n--- Calculated Class Weights ---")
    print(f"Weights for classes 0, 1, 2: {class_weights_dict}")

    # --- Define Model Architectures to Test ---
    configs = [
        {'name': 'Simpler', 'num_blocks': 2, 'initial_filters': 16, 'dense_units': 64},
        {'name': 'Current', 'num_blocks': 3, 'initial_filters': 16, 'dense_units': 128},
        {'name': 'Complex', 'num_blocks': 4, 'initial_filters': 16, 'dense_units': 256},
    ]

    results = []
    CLASS_NAMES = ['Healthy', 'Alzheimer\'s', 'FTD']

    # --- Main Experiment Loop ---
    for config in configs:
        print(f"\n{'='*25} TRAINING MODEL: {config['name']} {'='*25}")

        model = build_1d_cnn_eeg_model(
            input_shape=input_shape_for_model,
            num_classes=NUM_CLASSES,
            num_blocks=config['num_blocks'],
            initial_filters=config['initial_filters'],
            dense_units=config['dense_units']
        )
        model.summary()

        early_stopping_cb = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

        history = model.fit(X_train, y_train,
                             epochs=50,
                             batch_size=64,
                             validation_data=(X_val, y_val),
                             callbacks=[early_stopping_cb],
                             class_weight=class_weights_dict,
                             verbose=1
                            )

        # --- DETAILED EVALUATION (NOW CORRECTLY INSIDE THE LOOP) ---
        print(f"\n--- Detailed Evaluation for Model: {config['name']} ---")
        
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f"Confusion Matrix ({config['name']} Model)")
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"confusion_matrix_{config['name']}.png")
        plt.close()
        print(f"Confusion matrix saved to confusion_matrix_{config['name']}.png")

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
        plt.title(f"Accuracy ({config['name']} Model)")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.title(f"Loss ({config['name']} Model)")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"learning_curves_{config['name']}.png")
        plt.close()
        print(f"Learning curves plot saved to learning_curves_{config['name']}.png")

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        results.append({
            'name': config['name'],
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'val_accuracy': max(history.history['val_accuracy'])
        })

    # --- Print Final Comparative Results ---
    print(f"\n{'='*25} FINAL SUMMARY {'='*25}")
    for res in results:
        print(f"Model: {res['name']:<10} | Test Accuracy: {res['test_accuracy']:.4f} | Peak Validation Accuracy: {res['val_accuracy']:.4f}")


