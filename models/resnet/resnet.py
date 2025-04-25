import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold

# Constants
img_height = 224
img_width = 224
batch_size = 32
num_folds = 5  # Number of folds for cross-validation
epochs = 1

# Load the entire dataset (we'll split it ourselves for cross-validation)
full_ds = tf.keras.utils.image_dataset_from_directory(
    "chest_xray/train",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False  # We'll handle shuffling in KFold
)

# Get the file paths and labels for KFold splitting
file_paths = full_ds.file_paths
labels = np.concatenate([y for x, y in full_ds], axis=0)

# Convert to numpy arrays for KFold
images = np.concatenate([x for x, y in full_ds], axis=0)

# Define the KFold cross-validator
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)

# Store the history and scores for each fold
fold_no = 1
histories = []
accuracies = []
losses = []

for train_indices, val_indices in kfold.split(images, labels):
    print(f'\nTraining fold {fold_no}...')
    
    # Create training and validation datasets for this fold
    train_images, train_labels = images[train_indices], labels[train_indices]
    val_images, val_labels = images[val_indices], labels[val_indices]
    
    # Convert to TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_ds = train_ds.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create a new model for each fold
    model = tf.keras.applications.ResNet101(
        include_top=True,
        weights='imagenet',
        input_shape=(img_height, img_width, 3),
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )
    
    # Recompile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    
    # Store the history and metrics
    histories.append(history.history)
    accuracies.append(history.history['val_accuracy'][-1])
    losses.append(history.history['val_loss'][-1])
    
    fold_no += 1

# Print cross-validation results
print('\nCross-validation results:')
for i in range(num_folds):
    print(f'Fold {i+1}: Val Accuracy = {accuracies[i]:.4f}, Val Loss = {losses[i]:.4f}')

print(f'\nAverage accuracy across all folds: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})')
print(f'Average loss across all folds: {np.mean(losses):.4f} (+/- {np.std(losses):.4f})')