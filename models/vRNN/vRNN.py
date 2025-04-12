import os
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)
tf.random.set_seed(42)

# Constants
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)
BATCH_SIZE = 126
EPOCHS = 15
LATENT_DIM = 64

# Paths to dataset directories
BASE_DIR = "dataset/chest_xray"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Data augmentation and preprocessing for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load training data first to determine class indices
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

class_indices = train_generator.class_indices
class_names = list(class_indices.keys())
num_classes = len(class_names)

print(f"Detected classes: {class_names}")
print(f"Number of classes: {num_classes}")

# load validation and test data with the same class indices
validation_generator = test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    classes=class_names,
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")


# Define a custom Sampling layer to handle the reparameterization trick
class Sampling(Layer):

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# KL Loss Layer
class KLDivergenceLayer(Layer):
    """Layer for computing KL divergence loss"""

    def __init__(self, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # Compute KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1
        )

        # Add loss
        self.add_loss(kl_loss)

        # Return the inputs unchanged
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape


# VRNN Model
def build_vrnn_model(input_shape=INPUT_SHAPE, num_classes=num_classes):
    # Input layer
    inputs = Input(shape=input_shape)

    # CNN Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Reshape for RNN
    _, feature_h, feature_w, feature_c = x.shape
    x = Reshape((feature_h, feature_w * feature_c))(x)

    # Bidirectional LSTM (Encoder)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(256))(x)

    # Variational Latent Space
    z_mean = Dense(LATENT_DIM, name='z_mean')(x)
    z_log_var = Dense(LATENT_DIM, name='z_log_var')(x)

    # Apply KL divergence
    KLDivergenceLayer()([z_mean, z_log_var])

    # Use custom Sampling layer
    z = Sampling()([z_mean, z_log_var])

    # Decoder (Dense layers)
    decoder_h = Dense(256, activation='relu')(z)
    decoder_h = Dropout(0.5)(decoder_h)
    outputs = Dense(num_classes, activation='softmax')(decoder_h)

    # Define VRNN model
    model = Model(inputs, outputs)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Build and train VRNN
vrnn_model = build_vrnn_model()
vrnn_model.summary()

# Callbacks for training
callbacks = [
    ModelCheckpoint(
        'best_vrnn_chest_xray_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]

# Train the model
history = vrnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

# Load the best model
vrnn_model.load_weights('best_vrnn_chest_xray_model.keras')

# Evaluate the model on test data
test_loss, test_accuracy = vrnn_model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Make predictions on the test set
test_generator.reset()
y_pred = vrnn_model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true classes
y_true = test_generator.classes

# Display classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('vrnn_confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('vrnn_training_history.png')
plt.show()


# Function to predict a single image
def predict_image(image_path, model, class_names):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=IMAGE_SIZE
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale
    img_array = np.expand_dims(img_array, axis=0)  # Create batch

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]

    # Print results
    print(f"Predicted class: {predicted_class}")
    print("Class probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {predictions[0][i]:.4f}")

    return predicted_class, predictions[0]


# Save the model architecture and weights
vrnn_model.save('vrnn_chest_xray_model.keras')
print("Model saved to 'vrnn_chest_xray_model.keras'")


# Generate latent space visualization function
def generate_latent_visualization(model, generator, class_names, num_samples=100):
    # Get a batch of images
    generator.reset()
    batch_x, batch_y = next(generator)
    batch_x = batch_x[:num_samples]
    batch_y = batch_y[:num_samples]

    # Create a model that outputs the mean of latent space
    latent_model = Model(model.input, model.get_layer('z_mean').output)
    latent_vectors = latent_model.predict(batch_x)

    # Reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)

    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']  # Add more colors if needed

    for i, class_idx in enumerate(range(num_classes)):
        mask = np.argmax(batch_y, axis=1) == class_idx
        plt.scatter(
            latent_2d[mask, 0],
            latent_2d[mask, 1],
            c=colors[i % len(colors)],
            label=class_names[class_idx],
            alpha=0.7
        )

    plt.legend()
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.savefig('vrnn_latent_space.png')
    plt.close()


# Generate latent space visualization if we have enough data
try:
    generate_latent_visualization(vrnn_model, test_generator, class_names)
    print("Latent space visualization generated.")
except Exception as e:
    print(f"Could not generate latent space visualization: {e}")

