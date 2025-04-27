import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === PARAMETERS ===
img_size = 224
patch_size = 16
num_classes = 3   # Normal, Bacterial Pneumonia, Viral Pneumonia
batch_size = 32
num_patches = (img_size // patch_size) ** 2
projection_dim = 64
transformer_layers = 8
num_heads = 4
mlp_dim = 128
train_dir = "../../../chest_xray_crossvalidation/balanced/chest_xray/train"
val_dir = "../../../chest_xray_crossvalidation/balanced/chest_xray/val"
test_dir = "../../../chest_xray_crossvalidation/balanced/chest_xray/test"


# === PREPARE DATASETS ===
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode='categorical',
    image_size=(img_size, img_size),   # <-- FORCE RESIZE
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    label_mode='categorical',
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode='categorical',
    image_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False
)

# Normalize images to [0, 1]
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# === PATCH CREATION ===
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# === PATCH ENCODER ===
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

# === CREATE VIT MODEL ===
def create_vit_classifier():
    inputs = layers.Input(shape=(img_size, img_size, 3))
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Add Transformer blocks
    for _ in range(transformer_layers):
        # Layer normalization
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Self-attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        # Skip connection
        x2 = layers.Add()([attention_output, encoded_patches])

        # Feed-forward network
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(mlp_dim, activation='gelu')(x3)
        x3 = layers.Dense(projection_dim)(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dense(mlp_dim, activation='gelu')(representation)
    outputs = layers.Dense(num_classes, activation='softmax')(representation)

    # Model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# === COMPILE MODEL ===
vit_model = create_vit_classifier()
vit_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === TRAIN MODEL ===
history = vit_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# === TEST MODEL ===
test_loss, test_acc = vit_model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2f}")
