import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import cv2
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_dir = './kaggle/input/rice_leaf_diseases/'
data_dir
import pathlib
data_dir=pathlib.Path(data_dir)
data_dir
list(data_dir.glob("*DSC*.jpg"))
bacteria=list(data_dir.glob("Bacterial leaf blight/*"))
len(bacteria)
PIL.Image.open(str(bacteria[0]))
brown=list(data_dir.glob("Brown spot/*"))
len(brown)
dict={"bacteria":list(data_dir.glob("Bacterial leaf blight/*")),"brown":list(data_dir.glob("Brown spot/*")),"smut":list(data_dir.glob("Leaf smut/*"))}
labels_dict = {
    'bacteria': 0,
    'brown': 1,
    'smut': 2,
   
}
str(dict["smut"][0])
img=cv2.imread(str((dict["smut"][0])))
cv2.resize(img,(180,180)).shape


X, y = [], []

for name, images in dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img,(180,180))
        X.append(resized_img)
        y.append(labels_dict[name])

y[:5]

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
len(X_test)
X_train_scaled = X_train / 255
X_test_scaled = X_test / 255


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

model.evaluate(X_test_scaled,y_test)

predictions = model.predict(X_test_scaled)
predictions
score = tf.nn.softmax(predictions[0])
np.argmax(score)

y_test[0]

data_augmentation = keras.Sequential(
  [

    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomFlip("horizontal")
  ]
)

plt.axis('off')
plt.imshow(X[0])

plt.axis('off')
plt.imshow(data_augmentation(X)[0].numpy().astype("uint8"))
num_classes = 3

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
              
model.fit(X_train_scaled, y_train, epochs=40)    

model.evaluate(X_test_scaled,y_test)


def predict_disease(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (180, 180))
    scaled_img = resized_img / 255
    expanded_img = np.expand_dims(scaled_img, axis=0)
    prediction = model.predict(expanded_img)
    score = tf.nn.softmax(prediction[0])
    return np.argmax(score)

