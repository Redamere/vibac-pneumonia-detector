import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from pathlib import Path
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
my_loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
glblr=1e-3
glbs=16
early_stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

def count_images_in_directory(directory):
    directory = Path(directory)
    total_images = 0
    print("\n📸 Image count per class:")
    for class_folder in sorted(directory.iterdir()):
        if class_folder.is_dir():
            num_images = len(list(class_folder.glob("*")))
            print(f" - {class_folder.name}: {num_images} images")
            total_images += num_images
    print(f"\n🔢 Total images: {total_images}\n")


# 1. Load images manually into arrays
def load_data_as_arrays(directory, target_size=(32, 32)):
    X = []
    y = []
    class_indices = {}
    image_paths_by_class = {}

    directory = Path(directory)
    class_folders = [f for f in directory.iterdir() if f.is_dir()]
    class_folders.sort()  # Important for consistent label ordering

    # Assign an index to each class
    for idx, class_folder in enumerate(class_folders):
        class_indices[class_folder.name] = idx
        image_paths_by_class[class_folder.name] = list(class_folder.glob("*"))

    # Count images
    count_images_in_directory(directory)
    print(f"Class mappings: {class_indices}")

    # Find minimum number of images across all classes
    min_count = min(len(img_list) for img_list in image_paths_by_class.values())
    print(f"\n🔎 Minimum images per class: {min_count}")

    # For each class, load up to min_count images
    for class_name, img_paths in image_paths_by_class.items():
        label = class_indices[class_name]
        img_paths = shuffle(img_paths, random_state=42)  # Shuffle for randomness
        selected_paths = img_paths[:min_count]

        for img_path in selected_paths:
            if img_path.is_file():
                try:
                    img = image.load_img(img_path, target_size=target_size)
                    img_array = image.img_to_array(img)
                    img_array = img_array / 255.0  # Normalize to [0,1]
                    X.append(img_array)
                    y.append(label)
                except Exception as e:
                    print(f"⚠️ Warning: Could not process file {img_path}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y


# function to load test data
def load_test_data(directory, target_size=(32, 32), batch_size=glbs, num_classes=3):
    # Load the dataset from directory
    test_dataset = image_dataset_from_directory(
        directory,
        image_size=target_size,
        batch_size=batch_size,
        shuffle=False  # Do not shuffle, we want consistent evaluation
    )

    # Normalize the images (rescale) and one-hot encode labels
    normalization_layer = layers.Rescaling(1./255)
    
    # Applying normalization and one-hot encoding in the map function
    test_dataset = test_dataset.map(
        lambda x, y: (normalization_layer(x), to_categorical(y, num_classes=num_classes))
    )
    
    return test_dataset

# 2. Transformer Encoder Layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 3. Transformer Model Creation
def create_transformer_model(input_shape, num_classes=1, patch_size=8, embed_dim=64, learning_rate=glblr):
    inputs = keras.Input(shape=input_shape)

    patches = layers.Conv2D(filters=embed_dim,
                            kernel_size=patch_size,
                            strides=patch_size,
                            padding='valid')(inputs)
    
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patches = layers.Reshape((num_patches, embed_dim))(patches)
    
    x = TransformerEncoder(embed_dim=embed_dim, num_heads=8, ff_dim=512)(patches)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"]
    )
    return model

# 4. K-Fold Cross Validation
def run_k_fold(X, y, n_splits=5, epochs=5, batch_size=glbs):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []

    fold_no = 1
    for train_idx, val_idx in kfold.split(X):
        print(f"\nTraining fold {fold_no}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # One-hot encode the labels for each fold
        y_train = to_categorical(y_train, num_classes=len(np.unique(y)))
        y_val = to_categorical(y_val, num_classes=len(np.unique(y)))

        model = create_transformer_model(input_shape=X_train.shape[1:], num_classes=len(np.unique(y)), learning_rate=glblr)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[lr_scheduler, early_stopper]
        )

        scores = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold_no} - Loss: {scores[0]:.4f} - Accuracy: {scores[1]:.4f}")
        results.append(scores)

        fold_no += 1

    return results

# 5. Full Execution
def main():

    data_directory = "dataset/dataset_crossval/chest_xray"
    X, y = load_data_as_arrays(data_directory, target_size=(32, 32))

    # Step 1: Run K-Fold Cross-Validation
    results = run_k_fold(X, y, n_splits=5, epochs=40)

    # Step 2: Train Final Model on Full Dataset
    print("\nTraining final model on all data...")

    # One-hot encode the labels for the final model
    y_one_hot = to_categorical(y, num_classes=len(np.unique(y)))

    final_model = create_transformer_model(input_shape=X.shape[1:], num_classes=len(np.unique(y)), learning_rate=glblr)
    final_model.fit(X, y_one_hot, epochs=150, batch_size=glbs, verbose=1,callbacks=[lr_scheduler, early_stopper])

    # Step 3: Load and Evaluate on Test Data
    test_directory = "dataset/dataset_crossval/dataset_test"  # <-- Adjust if your test set is elsewhere

    print("\nLoading and evaluating on test data...")
    test_data = load_test_data(test_directory, target_size=(32, 32), batch_size=glbs)

    test_loss, test_accuracy = final_model.evaluate(test_data, verbose=1)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return results, final_model

# Run it
if __name__ == "__main__":
    results, final_model = main()


    print("\nCross-Validation Results (loss, accuracy) per fold:")
    for i, res in enumerate(results, 1):
        print(f"Fold {i}: Loss = {res[0]:.4f}, Accuracy = {res[1]:.4f}")
    losses, accuracies = zip(*results)
    print(f"\nAverage Loss: {np.mean(losses):.4f}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
