import numpy as np
import pandas as pd
import os
import pathlib
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

data_dir = './kaggle/input/rice_leaf_diseases/'
data_dir = pathlib.Path(data_dir)
images = list(data_dir.glob('*DSC*.jpg'))

dict = {
    "bacteria": list(data_dir.glob("Bacterial leaf blight/*")),
    "brown": list(data_dir.glob("Brown spot/*")),
    "smut": list(data_dir.glob("Leaf smut/*"))}

labels_dict = {'bacteria': 0, 'brown': 1, 'smut': 2}

X, y = [], []
for name, imgs in dict.items():
    for img in imgs:
        img = cv2.imread(str(img))
        resized_img = cv2.resize(img, (180, 180))
        X.append(resized_img)
        y.append(labels_dict[name])

X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

# Model without data augmentation
num_classes = 3
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Without data augmentation: Loss: {loss}, Accuracy: {accuracy}")

# Number of correct predictions
num_correct_preds = sum(np.argmax(model.predict(X_test_scaled), axis=-1) == y_test)
print(f"Without augmentation: {num_correct_preds}/{len(y_test)} correct predictions")

data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomFlip("horizontal")
])

# Model with data augmentation
model = Sequential([
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=30)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"With data augmentation: Loss: {loss}, Accuracy: {accuracy}")

# Number of correct predictions
num_correct_preds = sum(np.argmax(model.predict(X_test_scaled), axis=-1) == y_test)
print(f"With augmentation: {num_correct_preds}/{len(y_test)} correct predictions")

# Summarize and visualize the model
model.summary()
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

def predict_disease(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (180, 180))
    scaled_img = resized_img / 255
    expanded_img = np.expand_dims(scaled_img, axis=0)
    prediction = model.predict(expanded_img)
    score = tf.nn.softmax(prediction[0])
    return np.argmax(score)

