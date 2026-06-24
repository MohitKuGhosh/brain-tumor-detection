# brain_tumor_binary_detection.py

import os
import itertools
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras import regularizers

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

print ('Modules loaded')
# DATASET SETUP AND  BINARY MAPPING
print("\n--- DATASET SETUP ---")
train_data_dir = input("Please paste the full path to your TRAINING folder: ").strip().strip('"')
test_data_dir = input("Please paste the full path to your TESTING folder: ").strip().strip('"')

def create_dataframe(data_dir):
    filepaths = []
    labels = []
    
    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if not os.path.isdir(foldpath):
            continue
            
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            
            # BINARY MAPPING LOGIC
            if fold.lower() in ['no tumor', 'notumor', 'no_tumor']:
                labels.append('Healthy')
            else:
                labels.append('Tumor')
                
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    return pd.concat([Fseries, Lseries], axis=1)

train_df = create_dataframe(train_data_dir)
ts_df = create_dataframe(test_data_dir)

# Split Validation and Test
valid_df, test_df = train_test_split(ts_df, train_size=0.5, shuffle=True, random_state=123)

print(f"\nTraining Set: {len(train_df)} images")
print(f"Validation Set: {len(valid_df)} images")
print(f"Test Set: {len(test_df)} images")

# PRE-PROCESSING AND AUGMENTATION
batch_size = 16
img_size = (64, 64)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

# Data Augmentation
# This rotates and flips images to force the model to learn features
tr_gen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
ts_gen = ImageDataGenerator()

train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)

# MODEL ARCHITECTURE (Fine-Tuning)
class_count = 2 

# Unfreeze the base model
base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape)
base_model.trainable = True # Let the model learn!

# Freeze the bottom layers, train the top layers
# This adapts the model to MRI scans without destroying the pre-trained knowledge
for layer in base_model.layers[:-20]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(), # Better than pooling='max' for this case
    BatchNormalization(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5), # Increased dropout to prevent overfitting
    Dense(class_count, activation='softmax')
])

# IMPROVEMENT 3: Lower Learning Rate for Fine-Tuning
model.compile(Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# TRAINING
epochs = 15 # Increased epochs slightly since learning rate is lower
print("\nStarting Training... (This might take a bit longer, but will be more accurate)")
history = model.fit(x=train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False)

# GRAPHS & EVALUATION
tr_acc = history.history['accuracy']
tr_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
Epochs = [i+1 for i in range(len(tr_acc))]

plt.figure(figsize=(20, 8))
plt.style.use('fivethirtyeight')

plt.subplot(1, 2, 1)
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Evaluation on Test Set
print("\n--- TEST SET RESULTS ---")
preds = model.predict(test_gen)
y_pred = np.argmax(preds, axis=1)
g_dict = test_gen.class_indices
classes = list(g_dict.keys())

# Confusion Matrix
cm = confusion_matrix(test_gen.classes, y_pred)
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Healthy vs Tumor)')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(test_gen.classes, y_pred, target_names=classes))

model.save('Brain_Tumor_Binary.h5')
print("Model saved as 'Brain_Tumor_Binary.h5'")

# TUMOR DETECTION (HEATMAP)
loaded_model = tf.keras.models.load_model('Brain_Tumor_Binary.h5', compile=False)
loaded_model.compile(Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- SINGLE IMAGE DETECTION ---")
image_path = input("Please paste the full path to a single MRI image for testing: ").strip().strip('"')

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    base_model = model.layers[0]
    grad_model = tf.keras.models.Model(
        [base_model.inputs],
        [base_model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    plt.figure(figsize=(6, 6))
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.title('Tumor Detection (Heatmap)')
    plt.show()

try:
    image = Image.open(image_path)
    img = image.resize(img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = loaded_model.predict(img_array)
    class_labels = sorted(classes)
    score = tf.nn.softmax(predictions[0])
    result = class_labels[tf.argmax(score)]

    print(f"Predicted Class: {result}")

    # EfficientNetB0 last conv layer name is usually 'top_activation'
    last_conv_layer_name = "top_activation" 
    heatmap = make_gradcam_heatmap(img_array, loaded_model, last_conv_layer_name)
    save_and_display_gradcam(image_path, heatmap)
except Exception as e:
    print(f"Could not load image: {e}")
