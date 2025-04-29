import os
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

"""
Using the dataset without splits that Iker posted, this creates train/val/test sets
that all have balanced distributions so that we get good training results.
Change the orig_root and new_root to wherever you have the images saved and where you want them saved.
"""


# == USER PARAMETERS ==
orig_root = "../chest_xray_crossvalidation/chest_xray/"      # your source folder with BACTERIA, NORMAL, VIRUS
new_root = "../chest_xray_crossvalidation/balanced/chest_xray" # where new train/val/test will be created
img_height, img_width = 224, 224            # adjust to your model’s expected input
test_frac = 0.2                             # fraction for test set 20%
val_frac  = 0.1                             # fraction for val set (10% of remaining)

CLASS_NAMES = ["NORMAL", "BACTERIA", "VIRUS"]  # note typo fix below

# == VALIDATE CLASS NAMES ==
# Ensure that folder names match exactly
CLASS_NAMES = [c for c in ["NORMAL","BACTERIA","VIRUS"]]

# == STEP 1: COPY ALL IMAGES INTO A TEMP DIRECTORY ==

with tempfile.TemporaryDirectory() as tmpdir:
    # recreate class subfolders in tmpdir and copy
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(tmpdir, cls), exist_ok=True)
        src_dir = os.path.join(orig_root, cls)
        for fname in os.listdir(src_dir):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(tmpdir, cls, fname)
            # avoid overwriting duplicates
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(fname)
                dst_path = os.path.join(tmpdir, cls, f"{base}_dup{ext}")
            shutil.copy(src_path, dst_path)

    # == STEP 2: LOAD ALL IMAGES + LABELS ==
    datagen = ImageDataGenerator(rescale=1./255)
    total_imgs = sum(len(files) for _, _, files in os.walk(tmpdir))
    generator = datagen.flow_from_directory(
        tmpdir,
        target_size=(img_height, img_width),
        batch_size=total_imgs,
        class_mode='categorical',
        shuffle=False
    )
    # Grab the batch
    X, y = next(generator) # X: array of images, y: one‐hot labels matrix

# == STEP 3: SPLIT INTO TRAIN+VAL AND TEST ==
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, # images, labels
    test_size=test_frac,
    stratify=y.argmax(axis=1), # change to just y, aka labels maybe?
    random_state=42
)

# == STEP 4: SPLIT TRAIN+VAL INTO TRAIN AND VAL ==
adjusted_val = val_frac / (1 - test_frac)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=adjusted_val,
    stratify=y_trainval.argmax(axis=1), # change to just y_train_val maybe?
    random_state=42
)

# == STEP 5: FUNCTION TO SAVE SAMPLES BACK TO DISK ==
def save_split(X_split, y_split, subset_name):
    for idx, (img_arr, label_vec) in enumerate(zip(X_split, y_split)):
        cls_idx = label_vec.argmax()
        cls_name = CLASS_NAMES[cls_idx]
        out_dir = os.path.join(new_root, subset_name, cls_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{subset_name}_{cls_name}_{idx}.jpg")
        plt.imsave(out_path, img_arr)

# == STEP 6: WRITE OUT IMAGES ==
for split_name, Xs, ys in [
    ("train", X_train, y_train),
    ("val",   X_val,   y_val),
    ("test",  X_test,  y_test),
]:
    save_split(Xs, ys, split_name)
    print(f"Saved {split_name}: {len(Xs)} images")

print("Re‑splitting complete!")


def analyze_dataset(directory,
                    title="Dataset Analysis",
                    img_height=224,
                    img_width=224,
                    samples_per_class=3,
                    save_plots=False,
                    output_dir="."):
    """
    Analyze dataset and optionally save distribution and sample-image plots.
    
    Args:
        directory (str): Path to dataset folder (one subfolder per class).
        title (str): Printed and plotted title.
        img_height, img_width (int): Size for loading sample images.
        samples_per_class (int): How many examples to display per class.
        save_plots (bool): If True, save figures to `output_dir` instead of showing.
        output_dir (str): Where to write plot files (will be created if needed).
    """
    # 1) Compute counts
    classes = sorted(
        [d for d in os.listdir(directory)
         if os.path.isdir(os.path.join(directory, d))]
    )
    counts = {c: len(os.listdir(os.path.join(directory, c))) for c in classes}
    total = sum(counts.values())

    # Text summary
    print(f"\n{title}\n{'='*len(title)}")
    for c, cnt in counts.items():
        print(f"{c:10s}: {cnt:5d} ({cnt/total:.1%})")

    # ensure output dir
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # 2) Bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(classes, [counts[c] for c in classes])
    plt.title(f"{title}\nClass Distribution")
    plt.xlabel("Class")
    plt.ylabel("Num Images")
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + total*0.005,
                 f"{h}\n{h/total:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    if save_plots:
        fname = f"{title.replace(' ','_')}_distribution.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"Saved distribution plot to {fname}")
    else:
        plt.show()

    # 3) Sample-image grid
    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory(
        directory,
        target_size=(img_height, img_width),
        batch_size=samples_per_class * len(classes),
        class_mode='categorical',
        shuffle=True
    )
    imgs, labels = next(gen)

    plt.figure(figsize=(samples_per_class*3, len(classes)*3))
    for ci, cls in enumerate(classes):
        idxs = np.where(labels[:, ci] == 1)[0][:samples_per_class]
        for j, ii in enumerate(idxs):
            plt.subplot(len(classes), samples_per_class,
                        ci * samples_per_class + j + 1)
            plt.imshow(imgs[ii])
            plt.title(cls)
            plt.axis("off")
    plt.suptitle(f"Sample Images — {title}", y=1.02)
    plt.tight_layout()

    if save_plots:
        fname = f"{title.replace(' ','_')}_samples.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"Saved sample-image grid to {fname}")
    else:
        plt.show()

# Analyze new balanced datasets
# Shows distribution of each split and sample images.
train_dir = "../chest_xray_crossvalidation/balanced/chest_xray/train"
val_dir = "../chest_xray_crossvalidation/balanced/chest_xray/val"
test_dir = "../chest_xray_crossvalidation/balanced/chest_xray/test"
output_dir = "../"

analyze_dataset(train_dir, "Training Set", img_height, img_width, samples_per_class=5, save_plots=True, output_dir=output_dir)
analyze_dataset(val_dir, "Validation Set", img_height, img_width, samples_per_class=5, save_plots=True, output_dir=output_dir)
analyze_dataset(test_dir, "Test Set", img_height, img_width, samples_per_class=5, save_plots=True, output_dir=output_dir)

