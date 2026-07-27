import os
import shutil
import time
import zipfile
import uuid
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

plt.style.use('ggplot')

def timing(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    if h == 0 and m == 0:
        return f'{s:.2f} seconds'
    elif h == 0:
        return f'{m}m {s:.2f}s'
    else:
        return f'{h}h {m}m {s:.2f}s'

# 1. Path Resolution
possible_paths = ['.', 'dataset', 'dataset/brain_tumor_dataset']

yes_path = None
no_path = None

for path_option in possible_paths:
    y_p = os.path.join(path_option, 'yes')
    n_p = os.path.join(path_option, 'no')
    if os.path.exists(y_p) and os.path.exists(n_p):
        yes_path = y_p
        no_path = n_p
        break

if yes_path is None or no_path is None:
    if os.path.exists('archive.zip'):
        with zipfile.ZipFile('archive.zip', 'r') as z:
            z.extractall('dataset')
        print("Extracted archive.zip to dataset/", flush=True)
        for path_option in possible_paths:
            y_p = os.path.join(path_option, 'yes')
            n_p = os.path.join(path_option, 'no')
            if os.path.exists(y_p) and os.path.exists(n_p):
                yes_path = y_p
                no_path = n_p
                break

if yes_path is None or no_path is None:
    raise FileNotFoundError("Could not locate 'yes' and 'no' folders in dataset.")

base_dir = 'tumorous_and_non_tumorous'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

# Check if preprocessed folders already exist
need_preprocessing = not (os.path.exists(train_dir) and os.path.exists(valid_dir) and os.path.exists(test_dir))

if need_preprocessing:
    # 2. Augmentation Function
    def augmented_data(file_dir, n_generated, save_to_dir):
        data_gen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.15,
            brightness_range=(0.3, 1.0),
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        os.makedirs(save_to_dir, exist_ok=True)

        for filename in os.listdir(file_dir):
            src_path = os.path.join(file_dir, filename)
            dst_path = os.path.join(save_to_dir, filename)
            if os.path.isfile(src_path) and not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)

        files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
        for filename in files:
            img_path = os.path.join(file_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_batch = image_rgb.reshape((1,) + image_rgb.shape)

                for i in range(n_generated):
                    augmented_batch = next(data_gen.flow(x=image_batch, batch_size=1))
                    augmented_image = augmented_batch[0].astype('uint8')
                    base_name = os.path.splitext(filename)[0]
                    augmented_filename = f"aug_{base_name}_{uuid.uuid4().hex[:8]}.jpg"
                    save_path = os.path.join(save_to_dir, augmented_filename)
                    cv2.imwrite(save_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

    augmented_data_path = 'augmented_data'
    aug_yes_dir = os.path.join(augmented_data_path, 'yes')
    aug_no_dir = os.path.join(augmented_data_path, 'no')

    n_gen_yes = 6
    n_gen_no = 7

    print("Augmenting data...", flush=True)
    augmented_data(file_dir=yes_path, n_generated=n_gen_yes, save_to_dir=aug_yes_dir)
    augmented_data(file_dir=no_path, n_generated=n_gen_no, save_to_dir=aug_no_dir)

    def crop_brain_contour(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thres = cv2.erode(thres, None, iterations=2)
        thres = cv2.dilate(thres, None, iterations=2)
        cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        if not cnts:
            return image

        c = max(cnts, key=cv2.contourArea)
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        cropped = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        if cropped.size == 0:
            return image
        return cropped

    file_paths = []
    labels = []

    for f in os.listdir(aug_yes_dir):
        file_paths.append(os.path.join(aug_yes_dir, f))
        labels.append('tumorous')

    for f in os.listdir(aug_no_dir):
        file_paths.append(os.path.join(aug_no_dir, f))
        labels.append('non_tumorous')

    file_paths, labels = shuffle(file_paths, labels, random_state=42)

    paths_train, paths_temp, y_train, y_temp = train_test_split(file_paths, labels, test_size=0.3, random_state=42)
    paths_val, paths_test, y_val, y_test = train_test_split(paths_temp, y_temp, test_size=0.5, random_state=42)

    for split_dir in [train_dir, test_dir, valid_dir]:
        os.makedirs(os.path.join(split_dir, 'tumorous'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'non_tumorous'), exist_ok=True)

    def process_and_save_stream(paths, labels, dest_base_dir):
        IMG_SIZE = (240, 240)
        for i, (path, label) in enumerate(zip(paths, labels)):
            img = cv2.imread(path)
            if img is None:
                continue
            cropped_img = crop_brain_contour(img)
            resized_img = cv2.resize(cropped_img, IMG_SIZE, interpolation=cv2.INTER_CUBIC)
            save_path = os.path.join(dest_base_dir, label, f"{label}_{i}.jpg")
            cv2.imwrite(save_path, resized_img)

    process_and_save_stream(paths_train, y_train, train_dir)
    process_and_save_stream(paths_val, y_val, valid_dir)
    process_and_save_stream(paths_test, y_test, test_dir)
else:
    print("Using existing preprocessed dataset from local directory ('tumorous_and_non_tumorous').", flush=True)

# 3. High-Performance Data Generators & Model Architecture
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    shear_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(240, 240),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

valid_generator = val_test_datagen.flow_from_directory(
    valid_dir,
    target_size=(240, 240),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(240, 240),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Base VGG19
base_model = VGG19(input_shape=(240, 240, 3), include_top=False, weights='imagenet')

# Freeze initial layers, fine-tune top Block 5 layers for feature adaptation
for layer in base_model.layers[:-5]:
    layer.trainable = False
for layer in base_model.layers[-5:]:
    layer.trainable = True

# Fast GlobalAveragePooling2D Head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.00005), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 4. Model Fine-Tuning
callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1, mode='min'),
    ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

EPOCHS = 20
print("\n--- Starting Ultra-Fast Model Training ---", flush=True)
start_train = time.time()

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print(f"Training finished in {timing(time.time() - start_train)}", flush=True)

# 5. Model Evaluation on Test Set
print("\n--- Evaluating Model on Test Set ---", flush=True)
test_loss, test_acc = model.evaluate(test_generator)
print(f"\nFinal Test Results -> Accuracy: {test_acc * 100:.2f}% | Loss: {test_loss:.4f}", flush=True)

# Save training history plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_performance.png')
print("Saved training_performance.png plot successfully.", flush=True)


import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.00005  # Lower learning rate (5e-5) to prevent destroying pre-trained weights

# 2. Enhanced Data Augmentation Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    shear_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    fill_mode='nearest'
)

# 3. Load Pre-trained Base Model (VGG19)
base_model = VGG19(
    input_shape=(240, 240, 3), 
    include_top=False, 
    weights='imagenet'
)

# 4. FINE-TUNING CORE: Freeze early layers & Unfreeze top convolutional block (Block 5)
for layer in base_model.layers[:-5]:
    layer.trainable = False  # Keep early edge/shape feature detectors frozen

for layer in base_model.layers[-5:]:
    layer.trainable = True   # Unfreeze Block 5 to adapt high-level features for MRI scans

# 5. Build Classification Head with Regularization
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)  # Increased dropout to 0.3 for improved generalizability
predictions = Dense(2, activation='softmax')(x)

# 6. Compile Model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 7. Training Callbacks for Optimal Convergence
callbacks = [
    EarlyStopping(
        monitor='val_loss', 
        patience=6, 
        restore_best_weights=True, 
        verbose=1, 
        mode='min'
    ),
    ModelCheckpoint(
        'model.h5', 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1, 
        mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6, 
        verbose=1
    )
]

# 8. Execute Fine-Tuning
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
) 