import numpy as np
import pandas as pd
import tensorflow as tf
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(directory_path, target_col='SepsisLabel'):
    print("Loading data from directory:", directory_path)
    
    # Get all PSV files in the directory
    file_pattern = os.path.join(directory_path, "*.psv")
    file_paths = glob.glob(file_pattern)
    
    print(f"Found {len(file_paths)} patient files")
    
    # Create empty lists to store sequences and labels
    sequences = []
    labels = []
    
    # Process each patient file
    for file_path in file_paths:
        # Load patient data (pipe-separated values)
        patient_df = pd.read_csv(file_path, sep='|')
        patient_df.groupby('patient_id')
        patient_df = patient_df.drop(columns=['SBP', 'DBP', 'patient_id'], errors='ignore')
        
        # Get feature columns (excluding target)
        feature_cols = [col for col in patient_df.columns if col != target_col]
        
        # Extract features
        patient_features = patient_df[feature_cols].values
        
        # If any time in the sequence the patient developed sepsis, label as positive
        if target_col in patient_df.columns:
            sepsis_label = int(patient_df[target_col].max())
            
        # Add to lists
        sequences.append(patient_features)
        labels.append(sepsis_label)
    
    print(f"Successfully loaded data for {len(sequences)} patients")
    
    # Find distribution of positive/negative cases
    positive_cases = sum(labels)
    print(f"Class distribution: {positive_cases} positive cases, {len(labels) - positive_cases} negative cases")
    
    # Calculate sequence length statistics
    sequence_lengths = [len(seq) for seq in sequences]
    print(f"Sequence length - min: {min(sequence_lengths)}, max: {max(sequence_lengths)}, mean: {np.mean(sequence_lengths)}")
    
    # Convert labels to numpy array
    y = np.array(labels)
    
    # Split into train and test sets - keeping sequences as is
    indices = np.arange(len(sequences))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train = [sequences[i] for i in train_indices]
    X_test = [sequences[i] for i in test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    # Create a scaler
    n_features = sequences[0].shape[1]
    scaler = StandardScaler()
    
    # Combine all training data for fitting the scaler
    all_train_data = np.vstack([seq for seq in X_train])
    scaler.fit(all_train_data)
    
    # Scale each sequence individually
    X_train_scaled = [scaler.transform(seq) for seq in X_train]
    X_test_scaled = [scaler.transform(seq) for seq in X_test]
    
    print(f"Training sequences: {len(X_train_scaled)}")
    print(f"Test sequences: {len(X_test_scaled)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, n_features

class VariableLengthBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = [self.X[i] for i in batch_indices]
        batch_y = self.y[batch_indices]
        
        # Pad sequences within this batch only
        max_length = max(len(seq) for seq in batch_X)
        n_features = batch_X[0].shape[1]
        
        # Create padded batch
        padded_batch = np.zeros((len(batch_X), max_length, n_features))
        for i, seq in enumerate(batch_X):
            padded_batch[i, :len(seq), :] = seq
            
        return padded_batch, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def lstm(n_features, lstm_units=64):
    """
    Build and compile an LSTM model that can handle variable-length sequences
    Using layer normalization instead of batch normalization
    """
    # Using masking layer to handle variable length inputs
    inputs = tf.keras.layers.Input(shape=(None, n_features))
    mask = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    
    # First LSTM layer with layer normalization
    x = tf.keras.layers.LSTM(lstm_units, return_sequences=True)(mask)
    x = tf.keras.layers.LayerNormalization()(x)  # Layer normalization instead
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Second LSTM layer with layer normalization
    x = tf.keras.layers.LSTM(lstm_units // 2, return_sequences=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)  # Layer normalization instead
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision')
        ]
    )
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=30):
    """
    Train the LSTM model with early stopping using variable length sequences
    """
    # Create data generators
    train_gen = VariableLengthBatchGenerator(X_train, y_train, batch_size=batch_size)
    test_gen = VariableLengthBatchGenerator(X_test, y_test, batch_size=batch_size, shuffle=False)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    )
    
    # Class weights to handle imbalanced data
    class_counts = np.bincount(y_train.astype(int))
    if len(class_counts) > 1:  # Make sure we have both classes
        n_neg, n_pos = class_counts
        total = n_neg + n_pos
        weight_for_0 = (1 / n_neg) * (total / 2)
        weight_for_1 = (1 / n_pos) * (total / 2)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        print(f"Class weights: {class_weight}")
    else:
        class_weight = None
        print("Warning: Only one class present in training data")
    
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=epochs,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=1
    )
    
    return history, train_gen, test_gen

def evaluate_model(model, test_gen, y_test, history, threshold=0.5):
    """
    Evaluate the trained model with variable length sequences
    """
    # Predict on test generator
    y_pred_proba = []
    for i in range(len(test_gen)):
        batch_x, _ = test_gen[i]
        batch_pred = model.predict(batch_x)
        y_pred_proba.extend(batch_pred)
    
    y_pred_proba = np.array(y_pred_proba).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    print("\nModel Evaluation:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC score
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('figs/confusion_matrix.png')
    plt.show()
    
    # Plot ROC curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('figs/roc_curve.png')
    plt.show()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.tight_layout()
    plt.savefig('figs/training_history.png')
    plt.show()

def predict_sepsis(model, scaler, new_patient_data, threshold=0.5):
    """
    Make predictions for new patient data with variable length sequences
    
    Parameters:
    model: Trained Keras model
    scaler: Fitted StandardScaler
    new_patient_data: New patient time series data (list of 2D arrays)
    threshold (float): Classification threshold
    
    Returns:
    predictions: Binary predictions
    probabilities: Predicted probabilities
    """
    # Scale each sequence
    scaled_data = [scaler.transform(patient_data) for patient_data in new_patient_data]
    
    # Create a small batch generator with just the new data
    n_patients = len(scaled_data)
    probabilities = []
    
    # Process each patient individually or in small batches
    for i in range(n_patients):
        # Convert to 3D array (batch_size=1, seq_length, n_features)
        patient_data = np.expand_dims(scaled_data[i], axis=0)
        prob = model.predict(patient_data)[0][0]
        probabilities.append(prob)
    
    probabilities = np.array(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    
    return predictions, probabilities

if __name__ == "__main__":
    # Directory path containing patient PSV files
    data_directory = "/Users/jackbuxton/Documents/Applied data science/Cwk/cleaned_dataset"
    
    # Load and preprocess data - no padding
    X_train, X_test, y_train, y_test, scaler, n_features = load_data(data_directory)
    
    # Build model for variable length sequences
    model = lstm(n_features=n_features)
    print(model.summary())
    
    # Train model with variable length sequences
    history, train_gen, test_gen = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    evaluate_model(model, test_gen, y_test, history)
    
    print("Training and evaluation complete!")