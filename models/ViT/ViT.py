from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Define K-Fold
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# To store validation results
val_accuracies = []

# Loop over folds
for fold, (train_idx, val_idx) in enumerate(kf.split(x_train)):
    print(f"\n🌀 Fold {fold+1}/{k}")

    # Split the data
    x_tr, x_val = x_train[train_idx], x_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    # Build a fresh model each time
    model = build_vit()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Train on training split
    model.fit(x_tr, y_tr, epochs=10, batch_size=16, verbose=0)

    # Evaluate on validation split
    val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
    print(f"✅ Fold {fold+1} Accuracy: {val_acc:.4f}")

    val_accuracies.append(val_acc)

# Report final cross-validated accuracy
mean_acc = np.mean(val_accuracies)
std_acc = np.std(val_accuracies)
print(f"\n🎯 Average Accuracy over {k} folds: {mean_acc:.4f} ± {std_acc:.4f}")
